"""Full-field stress scan with automatic charging stops.

The 140 supplied 5 m waypoints describe scan rows spaced by 10 m. Flying
all of them is about 25.77 km and cannot finish inside 90 minutes. This
plan scans at 20 m AGL (camera footprint about 47.4 m wide), keeps every
fourth row for 40 m spacing and alternates direction. The resulting route
is about 6.91 km including transitions from CS1 and to the final charger.

Battery is sampled at 1 Hz. After a completed scan leg, a low battery
causes the mission to fly horizontally to the nearest charging station at
20 m, land on its centre, wait while disarmed until charged, take off to
20 m and continue with the next scan leg.

This mission only scans and writes ``stress_area.json``. It never sprays.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from local_planner import boot_drone, brake, fly_to, land, takeoff
from skytrack_autonomy.core.lib.scheduling import ScheduleGroup


TAG = "[FULL_STRESS_SCAN]"
STRESS_AREA_PATH = Path("/root/.ros/captures/stress_area.json")

# Camera / survey configuration.
TAKEOFF_ALT_M = 3.0
SCAN_ALT_M = 20.0
SCAN_SPEED_M_S = 5.0
TRANSIT_SPEED_M_S = 5.0
# First-run guard. Set to None only after the two-strip validation succeeds.
MAX_SCAN_SEGMENTS: int | None = 2
MAP_HZ = 0.5
PIXEL_STRIDE = 4
CAMERA_FX = 269.968
CAMERA_FY = 269.968
CAMERA_CX = 320.0
CAMERA_CY = 240.0
CAMERA_FORWARD_M = 0.125
CAMERA_HEIGHT_AT_HOME_M = 0.217

# One selected row can be about 415 m. The check is continuous, while a
# diversion is started after the current scan leg completes.
BATTERY_SAMPLE_HZ = 1.0
LOW_BATTERY_PERCENT = 20.0
RESUME_BATTERY_PERCENT = 95.0
CHARGER_TRANSIT_ALT_M = SCAN_ALT_M

# HSV baseline supplied by SkyTrack. CameraSense.decode() returns RGB.
YELLOW_HSV_LOW = (15, 40, 40)
YELLOW_HSV_HIGH = (28, 255, 255)
CELL_M = 0.5
MIN_SAMPLES_PER_CELL = 2
MIN_YELLOW_RATIO = 0.5
MIN_AREA_M2 = 4.0
CONTOUR_SIMPLIFY_M = 0.75

# ENU: x=east, y=north, z=pad elevation.
CHARGING_STATIONS = [
    ("CS1", 0.00, 0.00, 0.0),
    ("CS2", 329.32, -234.73, -1.0),
    ("CS3", 356.43, -654.32, 0.0),
    ("CS4", 41.72, -582.67, -1.1),
]

# Optimised from the supplied 140 points. Each entry is one scan segment:
# ((start_east, start_north), (end_east, end_north)).
# Rows are 40 m apart and ordered north-to-south with alternating direction.
SCAN_SEGMENTS_ENU = [
    ((-12.48, -13.85), (54.48, -13.85)),
    ((-38.85, -33.85), (376.56, -33.85)),
    ((380.42, -73.85), (-30.58, -73.85)),
    ((-22.31, -113.85), (384.27, -113.85)),
    ((388.12, -153.85), (-14.04, -153.85)),
    ((-5.76, -193.85), (391.97, -193.85)),
    ((395.83, -233.85), (2.51, -233.85)),
    ((10.78, -273.85), (399.68, -273.85)),
    ((403.53, -313.85), (19.05, -313.85)),
    ((27.32, -353.85), (407.38, -353.85)),
    ((411.24, -393.85), (35.59, -393.85)),
    ((43.86, -433.85), (415.09, -433.85)),
    ((418.94, -473.85), (52.13, -473.85)),
    ((60.41, -513.85), (422.80, -513.85)),
    ((426.65, -553.85), (68.68, -553.85)),
    ((76.95, -593.85), (144.44, -593.85)),
    ((189.17, -593.85), (426.07, -593.85)),
    ((401.86, -633.85), (215.59, -633.85)),
    ((242.01, -673.85), (276.13, -673.85)),
]


class BatteryMonitor:
    """Sample battery cheaply at 1 Hz without blocking flight control."""

    name = "battery_monitor"

    def __init__(self) -> None:
        self.ctx = None
        self.handle = None
        self.percent = None
        self.low = False

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx
        self.handle = ctx.scheduler.schedule(
            self._sample,
            hz=BATTERY_SAMPLE_HZ,
            group=ScheduleGroup.MEDIA,
            name=self.name,
            now=ctx.world.now(),
        )

    def _sample(self) -> None:
        if self.ctx is None:
            return
        value = self.ctx.senses.battery.percent
        self.percent = value
        was_low = self.low
        self.low = value is not None and value < LOW_BATTERY_PERCENT
        if self.low and not was_low:
            self.ctx.world.log_warn(
                f"{TAG} battery reached {value:.1f}%; charging required"
            )

    def shutdown(self) -> None:
        if self.ctx is not None and self.handle is not None:
            self.ctx.scheduler.unschedule(self.handle)
        self.handle = None
        self.ctx = None


class StressMap:
    """Accumulate yellow/seen votes in a fixed world-ENU ground grid."""

    def __init__(self) -> None:
        points = [point for segment in SCAN_SEGMENTS_ENU for point in segment]
        margin = 35.0
        self.x0 = min(p[0] for p in points) - margin
        self.y0 = min(p[1] for p in points) - margin
        x1 = max(p[0] for p in points) + margin
        y1 = max(p[1] for p in points) + margin
        rows = int(math.ceil((y1 - self.y0) / CELL_M)) + 1
        cols = int(math.ceil((x1 - self.x0) / CELL_M)) + 1
        self.seen = np.zeros((rows, cols), dtype=np.uint16)
        self.yellow = np.zeros((rows, cols), dtype=np.uint16)
        self.frames = 0
        self._last_seq = -1

    def sample(self, ctx: Any) -> None:
        """Decode the newest RGB frame directly; do no disk I/O."""
        camera = ctx.senses.camera
        if not camera.has_frame or camera.seq == self._last_seq:
            return
        pose = ctx.senses.pose.current_position
        if pose is None:
            return
        rgb = camera.decode()
        if rgb is None:
            return
        self.add_rgb(rgb, pose)
        self._last_seq = camera.seq
        self.frames += 1

    def add_rgb(self, rgb: np.ndarray, pose: Any) -> None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH) > 0
        height_px, width_px = yellow_mask.shape
        v, u = np.mgrid[0:height_px:PIXEL_STRIDE, 0:width_px:PIXEL_STRIDE]

        camera_height = max(0.1, -pose.z + CAMERA_HEIGHT_AT_HOME_M)
        fx = CAMERA_FX * width_px / 640.0
        fy = CAMERA_FY * height_px / 480.0
        cx = CAMERA_CX * width_px / 640.0
        cy = CAMERA_CY * height_px / 480.0
        right = (u - cx) / fx * camera_height
        back = (v - cy) / fy * camera_height

        cos_heading = math.cos(pose.heading)
        sin_heading = math.sin(pose.heading)
        camera_east = pose.y + CAMERA_FORWARD_M * sin_heading
        camera_north = pose.x + CAMERA_FORWARD_M * cos_heading
        east = camera_east + right * cos_heading - back * sin_heading
        north = camera_north - right * sin_heading - back * cos_heading

        row = ((north - self.y0) / CELL_M).astype(np.int32)
        col = ((east - self.x0) / CELL_M).astype(np.int32)
        inside = (
            (row >= 0)
            & (row < self.seen.shape[0])
            & (col >= 0)
            & (col < self.seen.shape[1])
        )
        row = row[inside]
        col = col[inside]
        np.add.at(self.seen, (row, col), 1)
        np.add.at(
            self.yellow,
            (row, col),
            yellow_mask[v, u][inside].astype(np.uint16),
        )

    def stress_areas(self) -> list[dict[str, Any]]:
        enough = self.seen >= MIN_SAMPLES_PER_CELL
        stressed = enough & (
            self.yellow >= np.ceil(self.seen * MIN_YELLOW_RATIO)
        )
        binary = stressed.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        areas = []
        epsilon_cells = CONTOUR_SIMPLIFY_M / CELL_M
        for contour in contours:
            area_m2 = cv2.contourArea(contour) * CELL_M * CELL_M
            if len(contour) < 3 or area_m2 < MIN_AREA_M2:
                continue
            contour = cv2.approxPolyDP(contour, epsilon_cells, True)
            if len(contour) < 3:
                continue
            polygon = [
                [
                    float(self.x0 + (col + 0.5) * CELL_M),
                    float(self.y0 + (row + 0.5) * CELL_M),
                ]
                for col, row in contour[:, 0]
            ]
            areas.append({"class": "stressed", "polygon": polygon})
        return areas


def save_stress_areas(areas: list[dict[str, Any]]) -> None:
    STRESS_AREA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STRESS_AREA_PATH.with_name(".stress_area.json.tmp")
    temporary.write_text(json.dumps({"stress_area": areas}, separators=(",", ":")))
    temporary.replace(STRESS_AREA_PATH)


def nearest_charger(pose: Any) -> tuple[str, float, float, float]:
    """Choose by horizontal distance; fly_to handles mapped obstacles."""
    return min(
        CHARGING_STATIONS,
        key=lambda station: math.hypot(pose.y - station[1], pose.x - station[2]),
    )


class WaitForRecharge:
    """Wait on the ground without publishing flight setpoints."""

    name = "wait_for_recharge"

    def __init__(self, station_name: str) -> None:
        self.station_name = station_name
        self.ctx: Any = None

    def start(self, ctx: Any, params: Any = None) -> None:
        self.ctx = ctx
        # LandSkill has already touched down and disarmed. Explicitly keep
        # offboard/setpoint control disabled while the simulator charges.
        ctx.world.publish_enable_to_fly(False)
        ctx.world.log_info(
            f"{TAG} landed at {self.station_name}; waiting for battery "
            f">= {RESUME_BATTERY_PERCENT:.0f}%"
        )

    def cancel(self, ctx: Any, reason: str) -> None:
        self.ctx = None

    @property
    def is_done(self) -> bool:
        if self.ctx is None:
            return False
        percent = self.ctx.senses.battery.percent
        return percent is not None and percent >= RESUME_BATTERY_PERCENT


def recharge_if_needed(ctx: Any, next_leg: int) -> Iterator[Any]:
    monitor = ctx.services.battery_monitor
    if not monitor.low:
        return

    pose = ctx.senses.pose.current_position
    if pose is None:
        ctx.world.log_warn(f"{TAG} battery low but pose unavailable")
        return
    station_name, east, north, _pad_z = nearest_charger(pose)
    ctx.world.log_warn(
        f"{TAG} battery={monitor.percent:.1f}%; checkpoint before leg "
        f"{next_leg + 1}, routing to {station_name}"
    )
    yield fly_to(
        north=north,
        east=east,
        alt_m=CHARGER_TRANSIT_ALT_M,
        target_speed=TRANSIT_SPEED_M_S,
        name=f"to_charge_{station_name}",
    )
    yield brake(name=f"settle_over_{station_name}")
    yield land(name=f"land_at_{station_name}")
    yield WaitForRecharge(station_name)
    yield takeoff(alt_m=SCAN_ALT_M)
    monitor.low = False
    ctx.world.log_info(
        f"{TAG} resumed from {station_name}; continuing at leg {next_leg + 1}"
    )


def full_field_stress_scan_mission(ctx: Any) -> Iterator[Any]:
    """Scan selected rows, checkpoint for charging, write stress polygons."""
    stress_map = StressMap()
    save_stress_areas([])  # remove stale detections from an earlier evaluation run
    yield takeoff(alt_m=TAKEOFF_ALT_M)

    active_segments = (
        SCAN_SEGMENTS_ENU
        if MAX_SCAN_SEGMENTS is None
        else SCAN_SEGMENTS_ENU[:MAX_SCAN_SEGMENTS]
    )
    ctx.world.log_info(
        f"{TAG} running {len(active_segments)}/{len(SCAN_SEGMENTS_ENU)} scan legs"
    )

    for index, (start, end) in enumerate(active_segments):
        yield from recharge_if_needed(ctx, index)

        yield fly_to(
            north=start[1],
            east=start[0],
            alt_m=SCAN_ALT_M,
            target_speed=TRANSIT_SPEED_M_S,
            name=f"leg_{index + 1:02d}_start",
        )

        mapper = ctx.scheduler.schedule(
            lambda: stress_map.sample(ctx),
            hz=MAP_HZ,
            group=ScheduleGroup.MEDIA,
            name=f"mapper_leg_{index + 1:02d}",
            now=ctx.world.now(),
        )
        try:
            yield fly_to(
                north=end[1],
                east=end[0],
                alt_m=SCAN_ALT_M,
                target_speed=SCAN_SPEED_M_S,
                mode="coverage",
                replan_mode="fast",
                name=f"leg_{index + 1:02d}_scan",
            )
        finally:
            ctx.scheduler.unschedule(mapper)

        areas = stress_map.stress_areas()
        save_stress_areas(areas)
        ctx.world.log_info(
            f"{TAG} leg {index + 1}/{len(active_segments)} complete; "
            f"frames={stress_map.frames}, areas={len(areas)}, "
            f"battery={ctx.senses.battery.percent}"
        )

    areas = stress_map.stress_areas()
    save_stress_areas(areas)
    ctx.world.log_info(
        f"{TAG} scan complete: {stress_map.frames} frames, "
        f"{len(areas)} stress area(s)"
    )

    pose = ctx.senses.pose.current_position
    station_name, east, north, _pad_z = nearest_charger(pose)
    yield fly_to(
        north=north,
        east=east,
        alt_m=CHARGER_TRANSIT_ALT_M,
        target_speed=TRANSIT_SPEED_M_S,
        name=f"final_return_{station_name}",
    )
    yield brake(name="pre_land")
    yield land()


full_field_stress_scan_mission.requires_senses = [
    "pose",
    "obstacle",
    "status",
    "battery",
    "camera",
]


def main() -> None:
    with boot_drone() as drone:
        drone.add_service(BatteryMonitor())
        drone.fly(full_field_stress_scan_mission)
        print(f"{TAG} services: {drone.list_services()}")
        drone.run()


if __name__ == "__main__":
    main()
