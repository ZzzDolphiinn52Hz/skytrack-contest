"""Mini scan video mission — record a downloadable MP4 while scanning.

SkyTrack's ``capture()`` saves still images inside the remote simulation
container.  This mission instead toggles the app recorder through its
``/local_planner/record`` ROS topic.  SkyTrack Desktop syncs the resulting
``rec_<timestamp>.mp4`` into the mission's local media/recordings folder.

Verify the scan rectangle is farmland and does not intersect a residential
area or its 5 m safety buffer before running.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Tuple

import rclpy
from std_msgs.msg import Bool

from local_planner import boot_drone, brake, fly_to, land, takeoff


NORTH_MIN_M = 0.0
NORTH_MAX_M = 12.0
EAST_MIN_M = 0.0
EAST_MAX_M = 8.0

TAKEOFF_ALT_M = 3.0
SCAN_ALT_M = 3.0
LINE_SPACING_M = 2.0
SCAN_SPEED_M_S = 1.0
TAG = "[MINI_SCAN_VIDEO]"


class DesktopRecorder:
    """Toggle the MP4 recorder exposed by the SkyTrack app adapter."""

    name = "desktop_recorder"

    def __init__(self) -> None:
        self._node = None
        self._publisher = None
        self._recording = False

    def attach(self, ctx: Any) -> None:
        # boot_drone() has already initialised rclpy before services attach.
        self._node = rclpy.create_node("mini_scan_video_recorder_control")
        self._publisher = self._node.create_publisher(
            Bool, "/local_planner/record", 10
        )

    def _publish(self, enabled: bool) -> None:
        if self._publisher is None:
            raise RuntimeError("DesktopRecorder used before attach()")
        message = Bool()
        message.data = enabled
        # Repeating the small control message makes the edge robust if DDS
        # discovery has only just completed. The method is non-blocking.
        for _ in range(3):
            self._publisher.publish(message)

    def start(self) -> bool:
        if self._recording:
            return False
        self._publish(True)
        self._recording = True
        return True

    def stop(self) -> bool:
        if not self._recording:
            return False
        self._publish(False)
        self._recording = False
        return True

    def shutdown(self) -> None:
        if self._recording and self._publisher is not None:
            self._publish(False)
            self._recording = False
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            self._publisher = None


def sweep_lines() -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return alternating (north, east) line endpoints."""
    if LINE_SPACING_M <= 0:
        raise ValueError("LINE_SPACING_M must be > 0")

    lines = []
    east = EAST_MIN_M
    reverse = False
    while east <= EAST_MAX_M + 1e-6:
        start = (NORTH_MIN_M, min(east, EAST_MAX_M))
        end = (NORTH_MAX_M, min(east, EAST_MAX_M))
        lines.append((end, start) if reverse else (start, end))
        east += LINE_SPACING_M
        reverse = not reverse
    return lines


def mini_scan_video_mission(ctx: Any) -> Iterator[Any]:
    """Record the camera while flying a small lawnmower scan."""
    lines = sweep_lines()
    recorder = ctx.services.desktop_recorder

    yield takeoff(alt_m=TAKEOFF_ALT_M)

    first = lines[0][0]
    yield fly_to(
        north=first[0],
        east=first[1],
        alt_m=SCAN_ALT_M,
        name="scan_entry",
    )
    yield brake(name="pre_record_settle")

    if not recorder.start():
        raise RuntimeError("SkyTrack desktop recorder did not start")
    ctx.world.log_info(f"{TAG} recorder ON; scanning {len(lines)} lines")

    for index, (start, end) in enumerate(lines, start=1):
        if index > 1:
            yield fly_to(
                north=start[0],
                east=start[1],
                alt_m=SCAN_ALT_M,
                name=f"line_{index}_start",
            )
        yield fly_to(
            north=end[0],
            east=end[1],
            alt_m=SCAN_ALT_M,
            mode="coverage",
            replan_mode="fast",
            target_speed=SCAN_SPEED_M_S,
            name=f"line_{index}_scan",
        )
        ctx.world.log_info(f"{TAG} line {index}/{len(lines)} complete")

    yield brake(name="post_scan_settle")
    recorder.stop()
    ctx.world.log_info(f"{TAG} recorder OFF")

    yield fly_to(north=0.0, east=0.0, alt_m=SCAN_ALT_M, name="return_cs1")
    yield brake(name="pre_land")
    yield land()


mini_scan_video_mission.requires_senses = [
    "pose",
    "obstacle",
    "status",
    "camera",
]


def main() -> None:
    with boot_drone() as drone:
        drone.add_service(DesktopRecorder())
        drone.fly(mini_scan_video_mission)
        print(f"{TAG} services: {drone.list_services()}")
        drone.run()


if __name__ == "__main__":
    main()
