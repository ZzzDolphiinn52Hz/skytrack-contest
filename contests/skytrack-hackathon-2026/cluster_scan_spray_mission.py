"""Scan one waypoint cluster at 5 m, detect stress, then spray at 5 m.

World coordinates supplied by the operator are ENU: x=east, y=north.
The drone first stages and fully charges at CS4, scans the eight-waypoint
lawnmower route, writes stress_area.json, sprays detected polygons, and
returns to CS4 to land. No Mission Break is used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from local_planner import boot_drone, brake, fly_to, land, takeoff
from skytrack_autonomy import Sprayer
from skytrack_autonomy.core.lib.scheduling import ScheduleGroup


TAG = "[CLUSTER_SCAN_SPRAY]"
STRESS_AREA_PATH = Path("/root/.ros/captures/stress_area.json")

# Replace this list when a new cluster is supplied. All points are flown at
# SCAN_ALT_M; the z values drawn in the SkyTrack UI are intentionally ignored.
SCAN_WAYPOINTS_ENU = [
    (98.07, -494.97),
    (157.25, -492.68),
    (157.65, -513.63),
    (99.02, -512.93),
    (98.68, -529.52),
    (157.05, -528.54),
    (157.05, -546.95),
    (99.08, -545.85),
]

# This cluster is closest to CS4.
CHARGER_NAME = "CS4"
CHARGER_EAST_M = 41.72
CHARGER_NORTH_M = -582.67
TRANSIT_ALT_M = 20.0
SCAN_ALT_M = 5.0
SPRAY_ALT_M = 5.0
FLIGHT_SPEED_M_S = 5.0
MAP_HZ = 1.0

# Automatic ground charging, verified by mission_break_charge_cycle_test.py.
CHARGED_PERCENT = 99.0
CHARGE_TIMEOUT_S = 180.0

# Stress detection, matching hackathon_example_v1.py.
YELLOW_HSV_LOW = (15, 40, 40)
YELLOW_HSV_HIGH = (28, 255, 255)
CELL_M = 0.2
MIN_SAMPLES_PER_CELL = 2
MIN_AREA_M2 = 2.0
CAMERA_HFOV_RAD = 1.74
CAMERA_HEIGHT_AT_HOME_M = 0.2

# Keep detections inside the waypoint cluster rather than spraying yellow
# roads/terrain visible beyond its boundary.
AOI_MIN_EAST = min(p[0] for p in SCAN_WAYPOINTS_ENU)
AOI_MAX_EAST = max(p[0] for p in SCAN_WAYPOINTS_ENU)
AOI_MIN_NORTH = min(p[1] for p in SCAN_WAYPOINTS_ENU)
AOI_MAX_NORTH = max(p[1] for p in SCAN_WAYPOINTS_ENU)

SPRAY_CONE_DEG = 30.0
SPRAY_SWATH_M = 2.0 * SPRAY_ALT_M * math.tan(
    math.radians(SPRAY_CONE_DEG / 2.0)
)
SPRAY_LANE_SPACING_M = SPRAY_SWATH_M * 0.85
MIN_SPRAY_RUN_M = 0.5
ENABLE_SPRAY = True


class WaitForGroundCharge:
    """Wait disarmed on a charging pad without flight setpoints."""

    name = "wait_for_ground_charge"

    def __init__(self, station_name: str) -> None:
        self.station_name = station_name
        self.ctx: Any = None
        self.started_at: float | None = None
        self.next_log_at: float | None = None
        self.charged = False

    def start(self, ctx: Any, params: Any = None) -> None:
        self.ctx = ctx
        self.started_at = ctx.world.now()
        self.next_log_at = self.started_at
        ctx.world.publish_enable_to_fly(False)
        ctx.world.log_info(f"{TAG} landed at {self.station_name}; charging")

    def cancel(self, ctx: Any, reason: str) -> None:
        self.ctx = None

    @property
    def is_done(self) -> bool:
        if self.ctx is None or self.started_at is None:
            return False
        now = self.ctx.world.now()
        percent = self.ctx.senses.battery.percent
        if self.next_log_at is not None and now >= self.next_log_at:
            self.ctx.world.log_info(f"{TAG} battery={percent}")
            self.next_log_at = now + 5.0
        if percent is not None and percent >= CHARGED_PERCENT:
            self.charged = True
            return True
        return now - self.started_at >= CHARGE_TIMEOUT_S


class StressMap:
    """Accumulate yellow/seen votes in a world-ENU ground grid."""

    def __init__(self) -> None:
        margin = 10.0
        self.x0 = AOI_MIN_EAST - margin
        self.y0 = AOI_MIN_NORTH - margin
        rows = int((AOI_MAX_NORTH + margin - self.y0) / CELL_M) + 1
        cols = int((AOI_MAX_EAST + margin - self.x0) / CELL_M) + 1
        self.seen = np.zeros((rows, cols), dtype=np.int32)
        self.yellow = np.zeros((rows, cols), dtype=np.int32)
        self.frames = 0
        self._last_seq = -1

    def snap(self, ctx: Any) -> None:
        camera = ctx.senses.camera
        if not camera.has_frame or camera.seq == self._last_seq:
            return
        pose = ctx.senses.pose.current_position
        if pose is None:
            return
        rgb = camera.decode()
        if rgb is None:
            return
        self.add_picture(rgb, pose)
        self._last_seq = camera.seq
        self.frames += 1

    def add_picture(self, rgb: np.ndarray, pose: Any) -> None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        is_yellow = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH) > 0
        h, w = is_yellow.shape
        v, u = np.mgrid[0:h:2, 0:w:2]
        focal = w / (2.0 * math.tan(CAMERA_HFOV_RAD / 2.0))
        height = max(0.1, -pose.z + CAMERA_HEIGHT_AT_HOME_M)
        right = (u - w / 2.0) / focal * height
        back = (v - h / 2.0) / focal * height
        cos_h, sin_h = math.cos(pose.heading), math.sin(pose.heading)
        east = pose.y + right * cos_h - back * sin_h
        north = pose.x - right * sin_h - back * cos_h
        row = ((north - self.y0) / CELL_M).astype(int)
        col = ((east - self.x0) / CELL_M).astype(int)
        inside = (
            (row >= 0) & (row < self.seen.shape[0])
            & (col >= 0) & (col < self.seen.shape[1])
        )
        row, col = row[inside], col[inside]
        np.add.at(self.seen, (row, col), 1)
        np.add.at(
            self.yellow,
            (row, col),
            is_yellow[v, u][inside].astype(np.int32),
        )

    def stress_areas(self) -> list[dict[str, Any]]:
        stressed = (
            (self.yellow * 2 >= self.seen)
            & (self.seen >= MIN_SAMPLES_PER_CELL)
        )

        # Strictly mask the output to the cluster bounding box.
        rows, cols = np.indices(stressed.shape)
        east = self.x0 + (cols + 0.5) * CELL_M
        north = self.y0 + (rows + 0.5) * CELL_M
        in_aoi = (
            (east >= AOI_MIN_EAST) & (east <= AOI_MAX_EAST)
            & (north >= AOI_MIN_NORTH) & (north <= AOI_MAX_NORTH)
        )
        binary = (stressed & in_aoi).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        areas = []
        for contour in contours:
            area_m2 = cv2.contourArea(contour) * CELL_M ** 2
            if len(contour) < 3 or area_m2 < MIN_AREA_M2:
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


def spray_lanes(
    polygon: list[list[float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    along_north = (
        max(p[1] for p in polygon) - min(p[1] for p in polygon)
        > max(p[0] for p in polygon) - min(p[0] for p in polygon)
    )
    points = [(p[1], p[0]) if along_north else (p[0], p[1]) for p in polygon]
    out = (
        (lambda along, across: (across, along))
        if along_north
        else (lambda along, across: (along, across))
    )
    low, high = min(p[1] for p in points), max(p[1] for p in points)
    count = max(1, math.ceil((high - low) / SPRAY_LANE_SPACING_M))
    step = (high - low) / count
    lanes = []
    for i in range(count):
        across = low + (i + 0.5) * step
        runs = [
            (a, b)
            for a, b in _inside_runs(points, across)
            if b - a >= MIN_SPRAY_RUN_M
        ]
        if i % 2:
            runs = [(b, a) for a, b in reversed(runs)]
        lanes.extend((out(a, across), out(b, across)) for a, b in runs)
    return lanes


def _inside_runs(
    points: list[tuple[float, float]], across: float
) -> list[tuple[float, float]]:
    hits = []
    for (aa, ac), (ba, bc) in zip(points, points[1:] + points[:1]):
        if (ac > across) == (bc > across):
            continue
        t = (across - ac) / (bc - ac)
        hits.append(aa + t * (ba - aa))
    hits.sort()
    return list(zip(hits[::2], hits[1::2]))


def cluster_scan_spray_mission(ctx: Any) -> Iterator[Any]:
    """Charge at CS4, scan the cluster, spray detections, return to CS4."""
    save_stress_areas([])

    # Long transit is flown at 20 m, then the battery is topped up at CS4.
    yield takeoff(alt_m=TRANSIT_ALT_M)
    yield fly_to(
        north=CHARGER_NORTH_M,
        east=CHARGER_EAST_M,
        alt_m=TRANSIT_ALT_M,
        target_speed=FLIGHT_SPEED_M_S,
        name=f"transit_to_{CHARGER_NAME}",
    )
    yield brake(name=f"settle_{CHARGER_NAME}")
    yield land(name=f"stage_land_{CHARGER_NAME}")
    charger = WaitForGroundCharge(CHARGER_NAME)
    yield charger
    if not charger.charged:
        ctx.world.log_warn(f"{TAG} charging timeout; stopping safely on pad")
        return

    stress_map = StressMap()
    yield takeoff(alt_m=SCAN_ALT_M)
    start_east, start_north = SCAN_WAYPOINTS_ENU[0]
    yield fly_to(
        north=start_north,
        east=start_east,
        alt_m=SCAN_ALT_M,
        target_speed=FLIGHT_SPEED_M_S,
        name="scan_start",
    )

    mapping = ctx.scheduler.schedule(
        lambda: stress_map.snap(ctx),
        hz=MAP_HZ,
        group=ScheduleGroup.MEDIA,
        name="cluster_mapper",
        now=ctx.world.now(),
    )
    try:
        for index, (east, north) in enumerate(SCAN_WAYPOINTS_ENU[1:], start=1):
            yield fly_to(
                north=north,
                east=east,
                alt_m=SCAN_ALT_M,
                target_speed=FLIGHT_SPEED_M_S,
                mode="coverage",
                name=f"scan_leg_{index:02d}",
            )
    finally:
        ctx.scheduler.unschedule(mapping)

    areas = stress_map.stress_areas()
    save_stress_areas(areas)
    ctx.world.log_info(
        f"{TAG} {stress_map.frames} frames -> {len(areas)} stress area(s)"
    )

    if ENABLE_SPRAY:
        sprayer = ctx.services.sprayer
        for area_index, area in enumerate(areas, start=1):
            lanes = spray_lanes(area["polygon"])
            for lane_index, (start, end) in enumerate(lanes, start=1):
                yield fly_to(
                    north=start[1], east=start[0], alt_m=SPRAY_ALT_M,
                    target_speed=FLIGHT_SPEED_M_S,
                    name=f"area_{area_index:02d}_lane_{lane_index:02d}_start",
                )
                ctx.world.log_info(
                    f"{TAG} spraying area {area_index}/{len(areas)}, "
                    f"lane {lane_index}/{len(lanes)}"
                )
                sprayer.on()
                try:
                    yield fly_to(
                        north=end[1], east=end[0], alt_m=SPRAY_ALT_M,
                        target_speed=FLIGHT_SPEED_M_S, mode="coverage",
                        name=f"area_{area_index:02d}_lane_{lane_index:02d}_spray",
                    )
                finally:
                    sprayer.off()

    yield fly_to(
        north=CHARGER_NORTH_M,
        east=CHARGER_EAST_M,
        alt_m=TRANSIT_ALT_M,
        target_speed=FLIGHT_SPEED_M_S,
        name=f"return_{CHARGER_NAME}",
    )
    yield brake(name="pre_land")
    yield land(name=f"final_land_{CHARGER_NAME}")


cluster_scan_spray_mission.requires_senses = [
    "pose", "obstacle", "status", "battery", "camera"
]


def main() -> None:
    with boot_drone() as drone:
        drone.add_service(Sprayer())
        drone.fly(cluster_scan_spray_mission)
        drone.run()


if __name__ == "__main__":
    main()
