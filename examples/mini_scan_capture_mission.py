"""Mini scan mission — collect a small georeferenced image dataset.

This is the first data-collection mission for the Sim Escape challenge.
It photographs a small rectangular patch using a boustrophedon
(lawnmower) route.  Each filename contains the grid row, column and the
commanded north/east position so that images can be matched to the
mission report later.

Safety
======
Verify the rectangle in SkyTrack Design view before every run.  It must
be farmland and must not intersect residential areas or their 5 m
buffer.  The default rectangle is deliberately small and matches the
area used by ``lawnmower_mission.py``; it is not a declaration that the
area is safe in every world revision.

Run::

    python -m local_planner.examples.mini_scan_capture_mission
"""
from __future__ import annotations

from typing import Any, Iterator, List, Tuple

from local_planner import CameraSense, boot_drone, brake, capture, fly_to, land, takeoff


# All horizontal coordinates are metres relative to the default spawn.
# API coordinates are (north, east); the world UI displays (x=east, y=north).
NORTH_MIN_M = 0.0
NORTH_MAX_M = 12.0
EAST_MIN_M = 0.0
EAST_MAX_M = 8.0

# Keep PX4 takeoff conservative, then climb to the imaging altitude with
# the normal trajectory planner.  Direct takeoff to 8 m was observed to
# stall with the X500 Nozzle System in SkyTrack Desktop 1.2.2.
TAKEOFF_ALT_M = 3.0
SCAN_ALT_M = 8.0
NORTH_SPACING_M = 3.0
EAST_SPACING_M = 4.0
TRANSIT_SPEED_M_S = 2.0
OUTPUT_DIR = "~/.ros/captures/mini_scan"
TAG = "[MINI_SCAN]"


PhotoPoint = Tuple[int, int, float, float]


def _axis_values(start: float, stop: float, spacing: float) -> List[float]:
    """Return inclusive, evenly spaced coordinates ending at ``stop``."""
    if stop < start:
        raise ValueError("scan bound stop must be >= start")
    if spacing <= 0:
        raise ValueError("scan spacing must be > 0")

    values: List[float] = []
    value = start
    while value <= stop + 1e-6:
        values.append(min(value, stop))
        value += spacing
    if values[-1] < stop - 1e-6:
        values.append(stop)
    return values


def photo_grid() -> List[PhotoPoint]:
    """Build a lawnmower-ordered photo grid over the scan rectangle."""
    north_values = _axis_values(NORTH_MIN_M, NORTH_MAX_M, NORTH_SPACING_M)
    east_values = _axis_values(EAST_MIN_M, EAST_MAX_M, EAST_SPACING_M)

    points: List[PhotoPoint] = []
    for row, east in enumerate(east_values):
        ordered_north = north_values if row % 2 == 0 else list(reversed(north_values))
        for visit_col, north in enumerate(ordered_north):
            original_col = (
                visit_col if row % 2 == 0 else len(north_values) - 1 - visit_col
            )
            points.append((row, original_col, north, east))
    return points


def mini_scan_capture_mission(ctx: Any) -> Iterator[Any]:
    """Fly the photo grid, capture sharp stills, return to CS1 and land."""
    points = photo_grid()
    ctx.world.log_info(
        f"{TAG} collecting {len(points)} images at {SCAN_ALT_M:.1f} m AGL"
    )

    yield takeoff(alt_m=TAKEOFF_ALT_M)

    for shot_index, (row, col, north, east) in enumerate(points, start=1):
        label = f"r{row:02d}_c{col:02d}"
        yield fly_to(
            north=north,
            east=east,
            alt_m=SCAN_ALT_M,
            mode="coverage",
            replan_mode="fast",
            target_speed=TRANSIT_SPEED_M_S,
            name=f"to_{label}",
        )
        yield brake(name=f"settle_{label}")

        # Encode commanded position in centimetres without '.', '-' or spaces.
        north_cm = round(north * 100)
        east_cm = round(east * 100)
        filename = (
            f"shot_{shot_index:03d}_{label}_n{north_cm:+06d}_e{east_cm:+06d}.png"
        )
        yield capture(
            output_dir=OUTPUT_DIR,
            filename=filename,
            timeout_s=5.0,
            name=f"photo_{label}",
        )
        ctx.world.log_info(
            f"{TAG} captured {shot_index}/{len(points)}: {filename}"
        )

    # CS1 is the default spawn: east=0, north=0.
    yield fly_to(north=0.0, east=0.0, alt_m=SCAN_ALT_M, name="return_cs1")
    yield brake(name="pre_land")
    yield land()


mini_scan_capture_mission.requires_senses = [
    "pose",
    "obstacle",
    "status",
    "camera",
]


def main() -> None:
    with boot_drone() as drone:
        drone.add_sense(CameraSense())
        drone.fly(mini_scan_capture_mission)
        print(f"{TAG} modes:  {drone.list_modes()}")
        print(f"{TAG} senses: {drone.list_senses()}")
        drone.run()


if __name__ == "__main__":
    main()
