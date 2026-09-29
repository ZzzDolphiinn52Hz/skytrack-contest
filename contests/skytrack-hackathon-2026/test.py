"""Two-stage whole-field mission: coarse 25 m scan, targeted 5 m scan, spray.

1. Screen the supplied crop polygon at 25 m using 15 m colour-context tiles.
2. Re-scan mottled/uncertain tiles and a small validation sample at 5 m.
3. Make the precise stress map, then spray its safe polygons at 3 m.
4. Route all transits around the three published no-fly polygons and land.

Coordinates are world ENU (x east, y north, z up). ``fly_to`` and the pose
are NED: ``fly_to(north=y, east=x)`` and ``east, north = pose.y, pose.x``.

Paste this file into the SkyTrack mission-script editor to run the mission.
"""
from __future__ import annotations

import json
import heapq
import math
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from local_planner import SkillStep, boot_drone, brake, fly_to, land, takeoff
from skytrack_autonomy import Sprayer
from skytrack_autonomy.core.lib.scheduling import ScheduleGroup

SURVEY_WAYPOINTS_ENU_1 = [            # (x, y, z), flown in order
    (17.2, -29.29, 5.5),
    (105.47, -34.16, 5.5),
]
SURVEY_WAYPOINTS_ENU_2= [            # (x, y, z), flown in order
    (83.46, -57.85, 5),
    (20.54, -56.38, 5),
]
SURVEY_WAYPOINTS_ENU_3 = [            # (x, y, z), flown in order
    (21.77, -75.40, 5.0),
    (78.75, -75.37, 5.0),
]
SURVEY_WAYPOINTS_ENU_4 = [            # (x, y, z), flown in order
    (82.61, -97.29, 5.5),
    (22.69, -97.39, 5.5),
]
SURVEY_WAYPOINTS_ENU_5 = [            # (x, y, z), flown in order
    (-2.68, -101.04, 5),
    (-4.44, -20.81, 5),
]
SURVEY_WAYPOINTS_ENU_6 = [            # (x, y, z), flown in order
    (-27.21, -30.98, 5),
    (-29.83, -10.95, 5),
    (-46.71, -12.58, 5.8),
    (-45.04, -29.92, 5.8),
]
SURVEY_WAYPOINTS_ENU_7 = [            # (x, y, z), flown in order
    (-47.42, -41.30, 5),
    (-21.19, -42.49, 5),
    (-21.62, -50.90, 5),
    (-47.67, -49.46, 5),
]
SURVEY_WAYPOINTS_ENU_8 = [            # (x, y, z), flown in order
    (-45.36, -61.88, 5.0),
    (-43.95, -69.73, 5.0),
    (-32.58, -69.69, 5.0),
    (-33.80, -61.45, 5.0),
    (-24.53, -60.96, 5.0),
    (-23.16, -68.80, 5.0),
]
SURVEY_WAYPOINTS_ENU_9 = [            # (x, y, z), flown in order
    (-22.26, -77.88, 5.0),
    (-45.22, -77.32, 5.0),
    (-45.06, -86.10, 5.0),
    (-18.88, -87.50, 5.0),
    (-19.18, -94.82, 5.0),
    (-45.19, -93.53, 5.0),
    (-45.07, -100.75, 5.0),
    (-19.34, -102.32, 5.0),
]
SURVEY_WAYPOINTS_ENU_10 = [            # (x, y, z), flown in order
    (-22.34, -128.62, 5.0),
    (-3.43, -267.73, 5.0),
    (1.42, -313.32, 5.0),
    (45.71, -564.21, 5.0),
    (65.33, -560.06, 5.0),
    (31.84, -347.20, 5.0),
]
SURVEY_WAYPOINTS_ENU_11 = [            # (x, y, z), flown in order
    (101.65, -318.28, 5.0),
    (60.86, -317.08, 5.0),
    (37.22, -289.85, 5.0),
    (101.35, -293.11, 5.0),
]
SURVEY_WAYPOINTS_ENU_12 = [            # (x, y, z), flown in order
    (144.33, -283.64, 5.0),
    (209.30, -285.54, 5.0),
    (210.05, -309.34, 5.0),
    (142.57, -306.58, 5.0),
    (133.55, -352.14, 5.0),
    (206.07, -344.28, 6.0),
    (209.63, -401.00, 6.0),
    (118.74, -407.98, 6.0),
    (119.02, -449.37, 6.0),
    (211.43, -444.63, 6.0),
]
SURVEY_WAYPOINTS_ENU_13 = [            # (x, y, z), flown in order
    (246.62, -492.68, 6.0),
    (239.68, -614.30, 6.0),
    (196.01, -582.99, 6.0),
    (200.55, -493.51, 6.0),
]
SURVEY_WAYPOINTS_ENU_14 = [            # (x, y, z), flown in order
    (155.35, -493.35, 5.0),
    (106.18, -489.58, 5.0),
    (103.54, -515.59, 5.0),
    (152.93, -519.45, 5.0),
    (155.03, -548.60, 5.0),
    (100.93, -542.95, 5.0),
]
SURVEY_WAYPOINTS_ENU_15 = [            # (x, y, z), flown in order
    (199.71, -569.53, 5.0),
    (203.45, -493.56, 5.5),
    (243.62, -494.54, 5.5),
    (236.24, -618.36, 5.5),
    (268.89, -614.67, 5.5),
    (274.60, -498.37, 5.5),
]


# Add SURVEY_WAYPOINTS_ENU_4, _5, ... here later, then append them to
# this list. Every survey is scanned, detected and sprayed before the next
# survey begins. The drone returns home only after the whole list is done.
SURVEYS = [
    SURVEY_WAYPOINTS_ENU_1,
    SURVEY_WAYPOINTS_ENU_2,
    SURVEY_WAYPOINTS_ENU_3,
    SURVEY_WAYPOINTS_ENU_4,
    SURVEY_WAYPOINTS_ENU_5,
    SURVEY_WAYPOINTS_ENU_6,
    SURVEY_WAYPOINTS_ENU_7,
    SURVEY_WAYPOINTS_ENU_8,
    SURVEY_WAYPOINTS_ENU_9,
    SURVEY_WAYPOINTS_ENU_10,
    SURVEY_WAYPOINTS_ENU_11,
    SURVEY_WAYPOINTS_ENU_12,
    SURVEY_WAYPOINTS_ENU_13,
    SURVEY_WAYPOINTS_ENU_14,
    SURVEY_WAYPOINTS_ENU_15,
]

# Supplied crop boundary, in perimeter order. The UI's z=2 m is NOT the
# survey altitude: the screening pass below flies at COARSE_ALT_M=25 m.
AOI_BOUNDARY_WAYPOINTS_ENU = [
    (-11.09, -18.69, 2.0), (-53.85, -21.44, 2.0),
    (37.14, -566.11, 2.0), (167.72, -554.10, 2.0),
    (229.75, -635.80, 2.0), (286.88, -609.63, 2.0),
    (433.17, -586.10, 2.0), (376.41, -22.21, 2.0),
    (306.41, 2.63, 2.0), (214.74, -5.24, 2.0),
    (160.29, -31.22, 2.0), (15.70, -23.01, 2.0),
]
AOI_POLYGON_ENU = [(x, y) for x, y, _ in AOI_BOUNDARY_WAYPOINTS_ENU]
COARSE_ALT_M = 25.0
COARSE_SPEED_M_S = 8.0
COARSE_TILE_M = 15.0
COARSE_MICRO_M = 5.0
COARSE_MICROS_PER_TILE = int(COARSE_TILE_M / COARSE_MICRO_M)
COARSE_LANE_SPACING_M = 40.0
COARSE_MIN_RUN_M = 15.0
AOI_FLIGHT_MARGIN_M = 4.0
FINE_ALT_M = 5.0
FINE_LANE_SPACING_M = 8.0
FINE_CLUSTER_TILES = 4  # at most 60 m x 60 m per 0.2 m stress map
COLD_CHECK_STRIDE = 20
COARSE_MAP_PATH = Path("/root/.ros/captures/coarse_candidate_map.json")

SURVEY_SPEED_M_S = 3.0
SURVEY_TRANSIT_SPEED_M_S = 12.0  # only between survey clusters, with mapper off
FAST_TRANSIT_MIN_CLEARANCE_M = 10.0  # slow down near residential buffers
FAST_TRANSIT_MIN_LEG_M = 15.0        # avoid high-speed short detour/corner legs
MAP_HZ = 1.0                        # pictures per second while surveying

# ── Stress detection ─────────────────────────────────────────────────
STRESS_AREA_PATH = Path("/root/.ros/captures/stress_area.json")   # attached to the report by the app
# OpenCV HSV: H 0-179, S/V 0-255. The calibration crops show green rice
# around H 34-37 and patchy yellow/brown rice extending roughly H 24-34.
GREEN_HSV_LOW = (34, 35, 145)
GREEN_HSV_HIGH = (70, 255, 255)
YELLOW_HSV_LOW = (20, 40, 125)
YELLOW_HSV_HIGH = (33, 255, 255)
BROWN_HSV_LOW = (8, 45, 125)       # light brown crop, not dark trees/ditches
BROWN_HSV_HIGH = (25, 255, 220)
DARK_VALUE_MAX = 120               # reject dark berms, ditches and tree crowns
DARK_CELL_RATIO = 0.50
DARK_BUFFER_M = 0.4                 # do not spray right against a dark berm
REFERENCE_VALUE_P75 = 175          # normalize camera exposure per frame
MIN_EXPOSURE_GAIN = 1.00           # never darken healthy green in a yellow frame
MAX_EXPOSURE_GAIN = 1.80
CELL_M = 0.2                        # ground map cell size
MIN_SAMPLES_PER_CELL = 2            # samples a cell needs to count
MIN_AREA_M2 = 2.0                   # smaller areas are dropped as noise
MIN_GREEN_COLOR_RATIO = 0.35        # healthy rice is mottled even when healthy
MIN_STRESS_COLOR_RATIO = 0.45       # per-cell dominant stress colour
MIN_GREEN_FRACTION = 0.12           # reject an almost uniformly yellow survey
MAX_BASELINE_YELLOW_FRACTION = 0.80 # reject a predominantly yellow parcel
GREEN_NEIGHBOR_RADIUS_M = 4.0       # yellow must border healthy green rice
MOTTLE_GROUP_GAP_M = 2.0           # group nearby yellow flecks into one patch
MIN_INTERIOR_GREEN_FRACTION = 0.03  # mottled stress has green inside its outline
MIN_INTERIOR_GREEN_CELLS = 4
GREEN_CONTEXT_MARGIN_M = 2.0       # or green must surround a solid yellow patch
MIN_GREEN_CONTEXT_SIDES = 3
MIN_SIDE_GREEN_FRACTION = 0.10
SURVEY_CORRIDOR_RADIUS_M = 5.0      # analyse only the flown field corridor
MAX_LINE_ARTIFACT_WIDTH_M = 1.5     # reject thin paths/field boundary lines
MIN_LINE_ARTIFACT_LENGTH_M = 4.0
CAMERA_HFOV_RAD = 1.74
CAMERA_HEIGHT_AT_HOME_M = 0.2       # camera height above ground before takeoff

# ── Spray ─────────────────────────────────────────────────────────────
SPRAY_ALT_M = 3.0
SPRAY_SPEED_M_S = 5.0
SPRAY_CONE_DEG = 30.0               # full cone angle
SPRAY_SWATH_M = 2.0 * SPRAY_ALT_M * math.tan(math.radians(SPRAY_CONE_DEG / 2.0))
SPRAY_LANE_SPACING_M = SPRAY_SWATH_M * 0.85     # 15% overlap between lanes
SPRAY_FOOTPRINT_MARGIN_M = 0.3     # centreline stays clear of healthy crop
MIN_LOCAL_STRESS_FRACTION = 0.75    # avoid spraying mostly healthy green
MIN_VISIBLE_FOOTPRINT_FRACTION = 0.45
MIN_SPRAY_RUN_M = 0.5               # shorter stretches are skipped
SPRAY_ENDPOINT_INSET_M = 1.0         # keep lane endpoints away from field edges
MIN_SPRAY_FLY_TIMEOUT_S = 20.0
SPRAY_FLY_TIMEOUT_FACTOR = 3.0
SPRAYER_COMMAND_TIMEOUT_S = 3.0

# ── Battery / charging ──────────────────────────────────────────────
BATTERY_CHECK_HZ = 1.0
LOW_BATTERY_PERCENT = 30.0
# PX4 commonly reports 99.x while the SkyTrack UI rounds it to 100%.
CHARGED_PERCENT = 99.0
CHARGE_TRANSIT_ALT_M = 20.0

# ENU: (name, east/x, north/y, pad elevation/z).
CHARGING_STATIONS = [
    ("CS1", 0.00, 0.00, 0.0),
    ("CS2", 329.32, -234.73, -1.0),
    ("CS3", 356.43, -654.32, 0.0),
    ("CS4", 41.72, -582.67, -1.1),
]

# Sim Escape semifinal no-fly-zones addendum, pages 2-3. All coordinates are
# world ENU (east, north), in the published vertex order. Treat them as
# forbidden at every altitude: climbing to 20 m does NOT permit crossing.
NO_FLY_ZONES_ENU = {
    1: [
        (322.520, -223.330), (313.757, -222.479), (305.961, -231.013),
        (307.462, -243.437), (322.615, -248.834), (326.240, -244.500),
        (325.120, -239.924), (324.990, -234.255), (325.411, -228.423),
    ],
    2: [
        (36.033, -596.647), (170.996, -572.019), (228.033, -653.620),
        (324.290, -643.696), (342.996, -658.942), (351.907, -678.162),
        (380.876, -665.190), (372.742, -641.293), (444.092, -601.030),
        (450.635, -626.219), (424.436, -660.437), (229.897, -714.460),
        (162.650, -612.493), (41.494, -632.769),
    ],
    3: [
        (377.969, 101.711), (385.962, 50.478), (372.808, -1.237),
        (366.570, -6.173), (348.243, -7.688), (341.196, 3.197),
        (334.850, 20.772), (312.815, 21.571), (248.884, 2.240),
        (235.677, 17.248), (205.699, 18.592), (198.071, -2.722),
        (181.145, -8.342), (148.987, -12.423), (135.177, -2.891),
        (125.547, -10.772), (110.886, -8.477), (107.682, -3.273),
        (98.105, -3.938), (94.562, -8.081), (87.159, -9.309),
        (68.483, -3.714), (62.533, 3.773), (60.982, 6.839),
        (50.878, 5.446), (43.325, 1.545), (40.471, -5.048),
        (32.504, -10.741), (14.419, -17.552), (9.170, -4.406),
        (6.992, 0.137), (0.277, 3.197), (-5.536, 0.619),
        (-6.088, -3.366), (-6.029, -9.157), (-10.118, -15.587),
        (-14.951, -13.513), (-23.984, -9.263), (-25.785, -3.495),
        (-69.843, -8.048), (-79.167, 47.698), (370.576, 102.435),
    ],
}
NO_FLY_CLEARANCE_M = 5.0  # mandatory residential buffer in the semifinal brief
NO_FLY_SPRAY_CLEARANCE_M = NO_FLY_CLEARANCE_M + 1.5  # cone radius stays outside it
PAD_MATCH_RADIUS_M = 1.0  # only an actual charger/spawn may use the exception
PAD_ACCESS_RADIUS_M = 4.0  # keep the exception local to the charging pad


def _cross(a: tuple[float, float], b: tuple[float, float],
           c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx
                            + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - start[0] - t * dx,
                      point[1] - start[1] - t * dy)


def _segments_intersect(a: tuple[float, float], b: tuple[float, float],
                        c: tuple[float, float], d: tuple[float, float]) -> bool:
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > 0 > ab_d or ab_d > 0 > ab_c)
            and (cd_a > 0 > cd_b or cd_b > 0 > cd_a)):
        return True
    eps = 1e-9
    for point, value, start, end in (
        (c, ab_c, a, b), (d, ab_d, a, b),
        (a, cd_a, c, d), (b, cd_b, c, d),
    ):
        if (abs(value) <= eps
                and min(start[0], end[0]) - eps <= point[0] <= max(start[0], end[0]) + eps
                and min(start[1], end[1]) - eps <= point[1] <= max(start[1], end[1]) + eps):
            return True
    return False


def _point_in_polygon(point: tuple[float, float],
                      polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for vertex in polygon:
        if ((previous[1] > y) != (vertex[1] > y)
                and x < (vertex[0] - previous[0]) * (y - previous[1])
                / (vertex[1] - previous[1]) + previous[0]):
            inside = not inside
        previous = vertex
    return inside


def _point_clear(point: tuple[float, float], margin_m: float) -> bool:
    for polygon in NO_FLY_ZONES_ENU.values():
        if _point_in_polygon(point, polygon):
            return False
        if any(_point_segment_distance(point, a, b) < margin_m - 1e-6
               for a, b in zip(polygon, polygon[1:] + polygon[:1])):
            return False
    return True


def _segment_clear(start: tuple[float, float], end: tuple[float, float],
                   margin_m: float = NO_FLY_CLEARANCE_M) -> bool:
    """Check the entire straight leg, not only its endpoints."""
    if not _point_clear(start, margin_m) or not _point_clear(end, margin_m):
        return False
    for polygon in NO_FLY_ZONES_ENU.values():
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            if _segments_intersect(start, end, a, b):
                return False
            if min(
                _point_segment_distance(start, a, b),
                _point_segment_distance(end, a, b),
                _point_segment_distance(a, start, end),
                _point_segment_distance(b, start, end),
            ) < margin_m - 1e-6:
                return False
    return True


def _nfz_corner_nodes(margin_m: float) -> list[tuple[float, float]]:
    """Offset obstacle convex corners into free space for a visibility graph."""
    nodes = []
    for polygon in NO_FLY_ZONES_ENU.values():
        signed_area = sum(a[0] * b[1] - b[0] * a[1]
                          for a, b in zip(polygon, polygon[1:] + polygon[:1]))
        winding = 1.0 if signed_area > 0 else -1.0
        for i, vertex in enumerate(polygon):
            previous, following = polygon[i - 1], polygon[(i + 1) % len(polygon)]
            if _cross(previous, vertex, following) * winding <= 1e-9:
                continue
            in_dx, in_dy = vertex[0] - previous[0], vertex[1] - previous[1]
            out_dx, out_dy = following[0] - vertex[0], following[1] - vertex[1]
            in_len, out_len = math.hypot(in_dx, in_dy), math.hypot(out_dx, out_dy)
            if min(in_len, out_len) < 1e-9:
                continue
            # A CCW obstacle has its free-space outward normal on the right.
            n1 = (winding * in_dy / in_len, -winding * in_dx / in_len)
            n2 = (winding * out_dy / out_len, -winding * out_dx / out_len)
            bisector = (n1[0] + n2[0], n1[1] + n2[1])
            bisector_len = math.hypot(*bisector)
            if bisector_len < 1e-9:
                continue
            direction = (bisector[0] / bisector_len, bisector[1] / bisector_len)
            projected = min(direction[0] * n1[0] + direction[1] * n1[1],
                            direction[0] * n2[0] + direction[1] * n2[1])
            if projected <= 1e-6:
                continue
            # Extra clearance allows for waypoint reach tolerance and tracking
            # error; the requested minimum clearance is still enforced below.
            base_offset = (margin_m + 1.0) / projected
            for scale in (1.0, 1.5, 2.0, 3.0):
                candidate = (vertex[0] + direction[0] * base_offset * scale,
                             vertex[1] + direction[1] * base_offset * scale)
                if _point_clear(candidate, margin_m + 0.01):
                    nodes.append(candidate)
                    break
    return nodes


def _pad_access_gate(
    point: tuple[float, float], margin_m: float,
) -> tuple[tuple[float, float], bool]:
    """Find a legal staging point for the brief's charger-pad exception."""
    if _point_clear(point, margin_m):
        return point, False
    if not _point_clear(point, 0.0):
        raise ValueError(f"Point lies inside a no-fly polygon: {point}")
    pads = [station for station in CHARGING_STATIONS
            if math.dist(point, (station[1], station[2])) <= PAD_MATCH_RADIUS_M]
    if not pads:
        raise ValueError(f"Point is inside the 5 m no-fly buffer, not at a pad: {point}")
    _, east, north, _ = min(
        pads, key=lambda station: math.dist(point, (station[1], station[2]))
    )
    pad = (east, north)

    nearest_distance = math.inf
    nearest_boundary = pad
    for polygon in NO_FLY_ZONES_ENU.values():
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length_sq = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((pad[0] - a[0]) * dx
                                    + (pad[1] - a[1]) * dy) / length_sq))
            boundary = (a[0] + t * dx, a[1] + t * dy)
            distance = math.dist(pad, boundary)
            if distance < nearest_distance:
                nearest_distance, nearest_boundary = distance, boundary
    if nearest_distance < 1e-6:
        raise ValueError(f"Charging pad touches a no-fly polygon: {pad}")

    direction = ((pad[0] - nearest_boundary[0]) / nearest_distance,
                 (pad[1] - nearest_boundary[1]) / nearest_distance)
    for extra in (1.0, 1.5, 2.0):
        offset = margin_m + extra - nearest_distance
        gate = (pad[0] + direction[0] * offset,
                pad[1] + direction[1] * offset)
        if (math.dist(pad, gate) <= PAD_ACCESS_RADIUS_M
                and _point_clear(gate, margin_m)
                and _segment_clear(point, gate, 0.0)):
            return gate, True
    raise RuntimeError(f"No safe local approach to charging pad: {pad}")


def _route_clear_enu(
    start: tuple[float, float], end: tuple[float, float],
    margin_m: float = NO_FLY_CLEARANCE_M,
) -> list[tuple[float, float]]:
    """Route strictly outside every polygon and its full buffer."""
    if not _point_clear(start, margin_m) or not _point_clear(end, margin_m):
        raise ValueError(f"No-fly clearance violated at route endpoint: {start} -> {end}")
    if _segment_clear(start, end, margin_m):
        return [end]

    nodes = [start, end] + _nfz_corner_nodes(margin_m)
    graph: list[list[tuple[float, int]]] = [[] for _ in nodes]
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if _segment_clear(a, b, margin_m):
                distance = math.dist(a, b)
                graph[i].append((distance, j))
                graph[j].append((distance, i))

    distances = [math.inf] * len(nodes)
    previous = [-1] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        if node == 1:
            break
        for edge_length, neighbour in graph[node]:
            candidate = distance + edge_length
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    if not math.isfinite(distances[1]):
        raise RuntimeError(f"No safe path around no-fly zones: {start} -> {end}")
    indices = []
    node = 1
    while node != 0:
        indices.append(node)
        node = previous[node]
    route = [nodes[i] for i in reversed(indices)]
    if not all(_segment_clear(a, b, margin_m)
               for a, b in zip([start] + route[:-1], route)):
        raise RuntimeError("No-fly routing generated an unsafe leg")
    return route


def safe_route_enu(
    start: tuple[float, float], end: tuple[float, float],
    margin_m: float = NO_FLY_CLEARANCE_M,
) -> list[tuple[float, float]]:
    """Route outside the 5 m buffer, except short legs to/from a charger."""
    if math.dist(start, end) < 1e-6:
        _pad_access_gate(start, margin_m)  # still reject unsafe non-pad points
        return [end]
    start_gate, start_at_pad = _pad_access_gate(start, margin_m)
    end_gate, end_at_pad = _pad_access_gate(end, margin_m)
    route = ([start_gate] if start_at_pad else [])
    route.extend(_route_clear_enu(start_gate, end_gate, margin_m))
    if end_at_pad:
        route.append(end)
    cleaned = []
    for point in route:
        if not cleaned or math.dist(point, cleaned[-1]) > 1e-6:
            cleaned.append(point)
    return cleaned


def safe_route_from_pose(ctx: Any, east: float, north: float) -> list[tuple[float, float]]:
    pose = ctx.senses.pose.current_position
    if pose is None:
        raise RuntimeError("Cannot plan a no-fly-safe path without current pose")
    return safe_route_enu((float(pose.y), float(pose.x)), (east, north))


def safe_fly_to(
    ctx: Any, *, east: float, north: float, alt_m: float,
    target_speed: float, name: str, mode: str = "coverage",
) -> Iterator[Any]:
    route = safe_route_from_pose(ctx, east, north)
    if len(route) > 1:
        ctx.world.log_info(f"[NFZ] {name}: routing via {len(route) - 1} safe detour(s)")
    pose = ctx.senses.pose.current_position
    previous = (float(pose.y), float(pose.x))
    for i, (point_east, point_north) in enumerate(route, start=1):
        point = (point_east, point_north)
        leg_speed = target_speed
        if (target_speed > SURVEY_SPEED_M_S
                and (math.dist(previous, point) < FAST_TRANSIT_MIN_LEG_M
                     or not _segment_clear(
                         previous, point, FAST_TRANSIT_MIN_CLEARANCE_M,
                     ))):
            leg_speed = SURVEY_SPEED_M_S
        yield fly_to(
            north=point_north, east=point_east, alt_m=alt_m,
            target_speed=leg_speed, mode=mode,
            name=name if i == len(route) else f"{name}_nfz_detour_{i:02d}",
        )
        previous = point


def preflight_no_fly_routes(
    home: tuple[float, float],
    surveys: list[list[tuple[float, float, float]]],
) -> int:
    """Reject unsafe fixed targets before takeoff; count planned detours."""
    for name, east, north, _ in CHARGING_STATIONS:
        try:
            _pad_access_gate((east, north), NO_FLY_CLEARANCE_M)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Charging station {name} has no safe pad approach") from exc
    previous = home
    detours = 0
    for survey_index, waypoints in enumerate(surveys, start=1):
        for waypoint_index, (east, north, _) in enumerate(waypoints, start=1):
            try:
                route = safe_route_enu(previous, (east, north))
            except (ValueError, RuntimeError) as exc:
                raise ValueError(
                    f"Survey {survey_index} waypoint {waypoint_index} has no safe route"
                ) from exc
            detours += len(route) - 1
            previous = (east, north)
    detours += len(safe_route_enu(previous, home)) - 1
    return detours


class BatteryMonitor:
    """Continuously sample battery at 1 Hz for recharge decisions."""

    name = "battery_monitor"

    def __init__(self) -> None:
        self.ctx: Any = None
        self.handle = None
        self.percent: float | None = None
        self.low = False

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx
        self.handle = ctx.scheduler.schedule(
            self._sample,
            hz=BATTERY_CHECK_HZ,
            group=ScheduleGroup.MEDIA,
            name=self.name,
            now=ctx.world.now(),
        )

    def _sample(self) -> None:
        if self.ctx is None:
            return
        percent = self.ctx.senses.battery.percent
        self.percent = percent
        was_low = self.low
        self.low = percent is not None and percent < LOW_BATTERY_PERCENT
        if self.low and not was_low:
            self.ctx.world.log_warn(
                f"[BATTERY] {percent:.1f}% < {LOW_BATTERY_PERCENT:.0f}%; "
                "recharge at the next safe boundary"
            )

    def shutdown(self) -> None:
        if self.ctx is not None and self.handle is not None:
            self.ctx.scheduler.unschedule(self.handle)
        self.handle = None
        self.ctx = None


class WaitForGroundCharge:
    """Wait disarmed on the pad until the UI reports a full battery."""

    name = "wait_for_ground_charge"

    def __init__(self, station_name: str) -> None:
        self.station_name = station_name
        self.ctx: Any = None
        self.next_log_at: float | None = None

    def start(self, ctx: Any, params: Any = None) -> None:
        self.ctx = ctx
        self.next_log_at = ctx.world.now()
        ctx.world.publish_enable_to_fly(False)
        ctx.world.log_info(
            f"[BATTERY] landed at {self.station_name}; waiting for 100%"
        )

    def cancel(self, ctx: Any, reason: str) -> None:
        self.ctx = None

    @property
    def is_done(self) -> bool:
        if self.ctx is None:
            return False
        now = self.ctx.world.now()
        percent = self.ctx.senses.battery.percent
        if self.next_log_at is not None and now >= self.next_log_at:
            self.ctx.world.log_info(
                f"[BATTERY] charging at {self.station_name}: {percent}%"
            )
            self.next_log_at = now + 5.0
        return percent is not None and percent >= CHARGED_PERCENT


def nearest_charger(pose: Any) -> tuple[str, float, float, float]:
    """Choose the charger with the shortest legal route, not straight range."""
    current = (float(pose.y), float(pose.x))

    def route_length(station: tuple[str, float, float, float]) -> float:
        route = safe_route_enu(current, (station[1], station[2]))
        return sum(math.dist(a, b)
                   for a, b in zip([current] + route[:-1], route))

    return min(
        CHARGING_STATIONS,
        key=route_length,
    )


def recharge_if_needed(ctx: Any, *, resume_alt_m: float) -> Iterator[Any]:
    """Land, charge, and take off; return True if recharge occurred."""
    monitor = ctx.services.battery_monitor
    if not monitor.low:
        return False

    pose = ctx.senses.pose.current_position
    if pose is None:
        ctx.world.log_warn("[BATTERY] low battery but pose is unavailable")
        return False

    station_name, east, north, _pad_z = nearest_charger(pose)
    ctx.world.log_warn(
        f"[BATTERY] {monitor.percent:.1f}%; diverting to {station_name}"
    )
    yield from safe_fly_to(
        ctx,
        north=north,
        east=east,
        alt_m=CHARGE_TRANSIT_ALT_M,
        target_speed=SURVEY_SPEED_M_S,
        name=f"recharge_transit_{station_name}",
    )
    yield brake(name=f"recharge_settle_{station_name}")
    yield land(name=f"recharge_land_{station_name}")
    yield WaitForGroundCharge(station_name)
    yield takeoff(alt_m=resume_alt_m)
    monitor.low = False
    ctx.world.log_info(
        f"[BATTERY] charged at {station_name}; resuming mission"
    )
    return True


def timed_fly_to(
    ctx: Any,
    *,
    north: float,
    east: float,
    alt_m: float,
    target_speed: float,
    name: str,
    mode: str | None = None,
) -> tuple[SkillStep, dict[str, bool]]:
    """Build a fly step that cannot hold forever on an unreachable target."""
    pose = ctx.senses.pose.current_position
    if pose is None:
        distance_m = 20.0
    else:
        distance_m = math.sqrt(
            (pose.x - north) ** 2
            + (pose.y - east) ** 2
            + ((-pose.z) - alt_m) ** 2
        )
    timeout_s = max(
        MIN_SPRAY_FLY_TIMEOUT_S,
        distance_m / max(0.5, target_speed) * SPRAY_FLY_TIMEOUT_FACTOR + 10.0,
    )
    deadline = ctx.world.now() + timeout_s
    result = {"timed_out": False}
    fly_options = {"mode": mode} if mode is not None else {}
    base = fly_to(
        north=north,
        east=east,
        alt_m=alt_m,
        target_speed=target_speed,
        replan_mode="fast",
        name=name,
        **fly_options,
    )

    def finished(c: Any) -> bool:
        if base.skill.is_done:
            return True
        if c.world.now() >= deadline:
            if not result["timed_out"]:
                result["timed_out"] = True
                c.world.log_warn(
                    f"[NAV] {name} timed out after {timeout_s:.1f}s; "
                    "skipping unreachable target"
                )
            return True
        return False

    return (
        SkillStep(skill=base.skill, is_done=finished, name=name),
        result,
    )


def wait_for_sprayer_condition(
    ctx: Any,
    *,
    condition: Any,
    name: str,
) -> tuple[SkillStep, dict[str, bool]]:
    """Hold briefly for an asynchronous sprayer state transition."""
    deadline = ctx.world.now() + SPRAYER_COMMAND_TIMEOUT_S
    result = {"timed_out": False}
    base = brake(name=name)

    def finished(c: Any) -> bool:
        if condition():
            return True
        if c.world.now() >= deadline:
            result["timed_out"] = True
            return True
        return False

    return (
        SkillStep(skill=base.skill, is_done=finished, name=name),
        result,
    )


def ensure_sprayer_open(
    ctx: Any,
    sprayer: Any,
    *,
    name: str,
) -> Iterator[Any]:
    """Wait for readiness, command ON, and return confirmed valve state."""
    if not sprayer.is_settled:
        settle_step, settle_result = wait_for_sprayer_condition(
            ctx,
            condition=lambda: sprayer.is_settled,
            name=f"{name}_wait_ready",
        )
        yield settle_step
        if settle_result["timed_out"]:
            ctx.world.log_warn(
                f"[SPRAY] {name}: payload did not settle; lane skipped"
            )
            return False

    accepted = sprayer.on()
    ctx.world.log_info(
        f"[SPRAY] {name}: ON accepted={accepted}, state={sprayer.state}, "
        f"settled={sprayer.is_settled}"
    )
    if accepted is False:
        ctx.world.log_warn(
            f"[SPRAY] {name}: ON command refused; lane skipped"
        )
        return False

    open_step, open_result = wait_for_sprayer_condition(
        ctx,
        condition=lambda: sprayer.state,
        name=f"{name}_wait_open",
    )
    yield open_step
    if open_result["timed_out"] or not sprayer.state:
        ctx.world.log_warn(
            f"[SPRAY] {name}: valve did not confirm open; lane skipped"
        )
        sprayer.off()
        return False

    ctx.world.log_info(f"[SPRAY] {name}: valve confirmed open")
    return True


def aoi_bounds() -> tuple[float, float, float, float]:
    """Bounding box of the supplied crop polygon, in world ENU metres."""
    return (
        min(p[0] for p in AOI_POLYGON_ENU), max(p[0] for p in AOI_POLYGON_ENU),
        min(p[1] for p in AOI_POLYGON_ENU), max(p[1] for p in AOI_POLYGON_ENU),
    )


def _point_inside_safe_aoi(point: tuple[float, float]) -> bool:
    if not _point_in_polygon(point, AOI_POLYGON_ENU):
        return False
    if any(_point_segment_distance(point, a, b) < AOI_FLIGHT_MARGIN_M
           for a, b in zip(AOI_POLYGON_ENU,
                           AOI_POLYGON_ENU[1:] + AOI_POLYGON_ENU[:1])):
        return False
    # Sampled waypoints are one metre apart; the extra metre guarantees
    # >=5 m no-fly clearance between samples as well as at them.
    return _point_clear(point, NO_FLY_CLEARANCE_M + 1.0)


def _safe_scan_runs(
    *, fixed: float, low: float, high: float, vertical: bool,
    minimum_m: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Clip a straight scan line to the crop AOI minus buffered no-fly zones."""
    samples = max(2, math.ceil(high - low) + 1)
    coordinates = np.linspace(low, high, samples)
    allowed = [
        _point_inside_safe_aoi((fixed, float(v)) if vertical else (float(v), fixed))
        for v in coordinates
    ]
    runs = []
    first = None
    for index, safe in enumerate(allowed + [False]):
        if safe and first is None:
            first = index
        elif not safe and first is not None:
            last = index - 1
            if float(coordinates[last] - coordinates[first]) >= minimum_m:
                a, b = float(coordinates[first]), float(coordinates[last])
                start, end = ((fixed, a), (fixed, b)) if vertical else ((a, fixed), (b, fixed))
                if not _segment_clear(start, end):
                    raise RuntimeError(f"Unsafe clipped scan leg: {start} -> {end}")
                runs.append((start, end))
            first = None
    return runs


def coarse_scan_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    x0, x1, y0, y1 = aoi_bounds()
    width = x1 - x0
    if width <= 40.0:
        xs = [0.5 * (x0 + x1)]
    else:
        count = max(2, math.ceil((width - 40.0) / COARSE_LANE_SPACING_M) + 1)
        xs = np.linspace(x0 + 20.0, x1 - 20.0, count).tolist()
    segments = []
    for index, x in enumerate(xs):
        runs = _safe_scan_runs(
            fixed=float(x), low=y0, high=y1, vertical=True,
            minimum_m=COARSE_MIN_RUN_M,
        )
        if index % 2 == 0:
            segments.extend((end, start) for start, end in reversed(runs))
        else:
            segments.extend(runs)
    if not segments:
        raise RuntimeError("No legal 25 m scan line inside the supplied AOI")
    return segments


class CoarseMap:
    """25 m screening map. Its 15 m tiles are NOT spray polygons."""

    def __init__(self) -> None:
        self.x0, self.x1, self.y0, self.y1 = aoi_bounds()
        self.cols = math.ceil((self.x1 - self.x0) / COARSE_TILE_M)
        self.rows = math.ceil((self.y1 - self.y0) / COARSE_TILE_M)
        self.micro_cols = math.ceil((self.x1 - self.x0) / COARSE_MICRO_M)
        self.micro_rows = math.ceil((self.y1 - self.y0) / COARSE_MICRO_M)
        shape = (self.rows, self.cols)
        self.seen = np.zeros(shape, dtype=np.int64)
        self.green = np.zeros(shape, dtype=np.int64)
        self.yellow = np.zeros(shape, dtype=np.int64)
        self.brown = np.zeros(shape, dtype=np.int64)
        micro_shape = (self.micro_rows, self.micro_cols)
        self.micro_seen = np.zeros(micro_shape, dtype=np.int64)
        self.micro_green = np.zeros(micro_shape, dtype=np.int64)
        self.micro_yellow = np.zeros(micro_shape, dtype=np.int64)
        self.micro_brown = np.zeros(micro_shape, dtype=np.int64)
        self.pictures = 0
        self._last_seq = -1
        self.micro_safe = np.zeros(micro_shape, dtype=bool)
        self.safe = np.zeros(shape, dtype=bool)
        for row in range(self.micro_rows):
            for col in range(self.micro_cols):
                xa = self.x0 + col * COARSE_MICRO_M
                xb = min(self.x1, xa + COARSE_MICRO_M)
                ya = self.y0 + row * COARSE_MICRO_M
                yb = min(self.y1, ya + COARSE_MICRO_M)
                centre = (0.5 * (xa + xb), 0.5 * (ya + yb))
                half_diagonal = 0.5 * math.hypot(xb - xa, yb - ya)
                inside_crop = (
                    _point_in_polygon(centre, AOI_POLYGON_ENU)
                    and all(_point_segment_distance(centre, a, b) >= half_diagonal
                            for a, b in zip(AOI_POLYGON_ENU,
                                            AOI_POLYGON_ENU[1:] + AOI_POLYGON_ENU[:1]))
                )
                if inside_crop and _point_clear(
                    centre, NO_FLY_CLEARANCE_M + half_diagonal,
                ):
                    self.micro_safe[row, col] = True
                    tile_row = min(self.rows - 1, row // COARSE_MICROS_PER_TILE)
                    tile_col = min(self.cols - 1, col // COARSE_MICROS_PER_TILE)
                    self.safe[tile_row, tile_col] = True

    def snap(self, ctx: Any) -> None:
        camera = ctx.senses.camera
        if not camera.has_frame or camera.seq == self._last_seq:
            return
        pose = ctx.senses.pose.current_position
        if pose is None:
            return
        rgb = camera.decode()  # SkyTrack /camera is rgb8; no cloud PNG write.
        if rgb is None:
            return
        self.add_picture(rgb, pose)
        self._last_seq = camera.seq
        self.pictures += 1

    def add_picture(self, rgb: np.ndarray, pose: Any) -> None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, w = hsv.shape[:2]
        reference = hsv[:max(1, int(0.47 * h))]
        reference_pixels = reference[:, :, 2][
            (reference[:, :, 1] >= 40) & (reference[:, :, 2] >= 50)
        ]
        if reference_pixels.size >= 100:
            gain = float(np.clip(
                REFERENCE_VALUE_P75 / max(1.0, float(np.percentile(reference_pixels, 75))),
                MIN_EXPOSURE_GAIN, MAX_EXPOSURE_GAIN,
            ))
            hsv[:, :, 2] = np.clip(
                hsv[:, :, 2].astype(np.float32) * gain, 0, 255,
            ).astype(np.uint8)
        green = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH) > 0
        brown = cv2.inRange(hsv, BROWN_HSV_LOW, BROWN_HSV_HIGH) > 0
        yellow = (cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH) > 0) & ~brown
        bright = hsv[:, :, 2] >= DARK_VALUE_MAX

        v, u = np.mgrid[0:h:4, 0:w:4]
        occluded = ((v >= 0.47 * h) & (
            ((u >= 0.15 * w) & (u <= 0.34 * w))
            | ((u >= 0.67 * w) & (u <= 0.85 * w))
        )) | ((v >= 0.75 * h) & (u >= 0.34 * w) & (u <= 0.67 * w))
        focal = w / (2.0 * math.tan(CAMERA_HFOV_RAD / 2.0))
        height = -pose.z + CAMERA_HEIGHT_AT_HOME_M
        right = (u - w / 2.0) / focal * height
        back = (v - h / 2.0) / focal * height
        cos_h, sin_h = math.cos(pose.heading), math.sin(pose.heading)
        east = pose.y + right * cos_h - back * sin_h
        north = pose.x - right * sin_h - back * cos_h
        col = np.floor((east - self.x0) / COARSE_TILE_M).astype(int)
        row = np.floor((north - self.y0) / COARSE_TILE_M).astype(int)
        micro_col = np.floor((east - self.x0) / COARSE_MICRO_M).astype(int)
        micro_row = np.floor((north - self.y0) / COARSE_MICRO_M).astype(int)
        inside = ((row >= 0) & (row < self.rows)
                  & (col >= 0) & (col < self.cols)
                  & (micro_row >= 0) & (micro_row < self.micro_rows)
                  & (micro_col >= 0) & (micro_col < self.micro_cols)
                  & ~occluded & bright[v, u])
        row, col = row[inside], col[inside]
        micro_row, micro_col = micro_row[inside], micro_col[inside]
        v, u = v[inside], u[inside]
        if row.size == 0:
            return
        allowed = self.micro_safe[micro_row, micro_col]
        row, col = row[allowed], col[allowed]
        micro_row, micro_col = micro_row[allowed], micro_col[allowed]
        v, u = v[allowed], u[allowed]
        flat = row * self.cols + col
        micro_flat = micro_row * self.micro_cols + micro_col
        colours = (
            (np.ones(flat.shape, dtype=bool), self.seen, self.micro_seen),
            (green[v, u], self.green, self.micro_green),
            (yellow[v, u], self.yellow, self.micro_yellow),
            (brown[v, u], self.brown, self.micro_brown),
        )
        for colour, tile_target, micro_target in colours:
            weights = colour.astype(np.int32)
            tile_target += np.bincount(
                flat, weights=weights, minlength=tile_target.size,
            ).reshape(tile_target.shape).astype(np.int64)
            micro_target += np.bincount(
                micro_flat, weights=weights, minlength=micro_target.size,
            ).reshape(micro_target.shape).astype(np.int64)

    def candidates(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Use 5 m subtiles to separate uniform colour from mottled crop."""
        labels = np.full((self.rows, self.cols), "excluded", dtype=object)
        records = []
        for row in range(self.rows):
            for col in range(self.cols):
                count = int(self.seen[row, col])
                crop = int(self.green[row, col] + self.yellow[row, col]
                           + self.brown[row, col])
                green_fraction = float(self.green[row, col]) / max(1, crop)
                stress_fraction = float(self.yellow[row, col]
                                        + self.brown[row, col]) / max(1, crop)
                yellow_fraction = float(self.yellow[row, col]) / max(1, crop)
                brown_fraction = float(self.brown[row, col]) / max(1, crop)
                micros = np.s_[
                    row * COARSE_MICROS_PER_TILE:
                    min(self.micro_rows, (row + 1) * COARSE_MICROS_PER_TILE),
                    col * COARSE_MICROS_PER_TILE:
                    min(self.micro_cols, (col + 1) * COARSE_MICROS_PER_TILE),
                ]
                micro_seen = self.micro_seen[micros]
                micro_crop = (self.micro_green[micros]
                              + self.micro_yellow[micros] + self.micro_brown[micros])
                valid_micro = ((micro_seen >= 20)
                               & (micro_crop >= 0.30 * np.maximum(1, micro_seen)))
                local_stress = (
                    (self.micro_yellow[micros] + self.micro_brown[micros])
                    / np.maximum(1, micro_crop)
                )[valid_micro]
                patch_peak = float(np.max(local_stress)) if local_stress.size else 0.0
                patch_spread = (float(np.max(local_stress) - np.min(local_stress))
                                if local_stress.size >= 2 else 0.0)
                if self.safe[row, col]:
                    if count < 40 or np.count_nonzero(valid_micro) < 2:
                        label = "uncertain"
                    elif crop / count < 0.30:
                        label = "noncrop"
                    elif (green_fraction >= 0.85 and patch_peak < 0.15
                          and patch_spread < 0.12):
                        label = "uniform_green"
                    elif (green_fraction <= 0.08 and yellow_fraction >= 0.80
                          and patch_spread < 0.12):
                        label = "uniform_yellow"
                    elif (green_fraction >= 0.15 and stress_fraction <= 0.85
                          and ((stress_fraction >= 0.18 and patch_spread >= 0.10)
                               or (stress_fraction >= 0.05 and patch_peak >= 0.28
                                   and patch_spread >= 0.18))):
                        label = "hot"
                    elif (green_fraction >= 0.08
                          and (stress_fraction >= 0.06 or patch_peak >= 0.20)):
                        label = "uncertain"
                    elif brown_fraction >= 0.25:
                        label = "uncertain"
                    else:
                        label = "cold"
                    labels[row, col] = label
                records.append({
                    "row": row, "col": col, "label": labels[row, col],
                    "center_enu": [
                        round(0.5 * (self.x0 + col * COARSE_TILE_M
                                     + min(self.x1, self.x0 + (col + 1) * COARSE_TILE_M)), 2),
                        round(0.5 * (self.y0 + row * COARSE_TILE_M
                                     + min(self.y1, self.y0 + (row + 1) * COARSE_TILE_M)), 2),
                    ],
                    "seen": count, "crop_fraction": round(crop / max(1, count), 3),
                    "green_fraction": round(green_fraction, 3),
                    "yellow_fraction": round(yellow_fraction, 3),
                    "brown_fraction": round(brown_fraction, 3),
                    "patch_peak": round(patch_peak, 3),
                    "patch_spread": round(patch_spread, 3),
                })
        selected = (labels == "hot") | (labels == "uncertain")
        # A one-tile halo catches patch edges, but strong uniform green/yellow
        # tiles remain skipped as requested.
        hot_neighbours = cv2.dilate(
            (labels == "hot").astype(np.uint8), np.ones((3, 3), np.uint8),
        ) > 0
        selected |= hot_neighbours & ((labels == "cold") | (labels == "uncertain"))
        # Only weakly classified negatives are sampled, not large uniform fields.
        cold = [(r, c) for r in range(self.rows) for c in range(self.cols)
                if labels[r, c] == "cold"]
        for index in range(0, len(cold), COLD_CHECK_STRIDE):
            selected[cold[index]] = True
        selected &= self.safe
        for record in records:
            record["fine_scan"] = bool(selected[record["row"], record["col"]])
        return selected, records


def fine_scan_segments(
    coarse: CoarseMap, selected: np.ndarray,
    current: tuple[float, float],
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    first_row, last_row = row_range or (0, coarse.rows)
    first_col, last_col = col_range or (0, coarse.cols)
    pending = []
    for row in range(first_row, last_row):
        col = first_col
        while col < last_col:
            if not selected[row, col]:
                col += 1
                continue
            first = col
            while col < last_col and selected[row, col]:
                col += 1
            xa = coarse.x0 + first * COARSE_TILE_M
            xb = min(coarse.x1, coarse.x0 + col * COARSE_TILE_M)
            ya = coarse.y0 + row * COARSE_TILE_M
            yb = min(coarse.y1, ya + COARSE_TILE_M)
            if yb - ya <= 0.0:
                continue
            if yb - ya <= 8.0:
                ys = [0.5 * (ya + yb)]
            else:
                lane_count = max(2, math.ceil((yb - ya - 8.0) / FINE_LANE_SPACING_M) + 1)
                ys = np.linspace(ya + 4.0, yb - 4.0, lane_count)
            for y in ys:
                pending.extend(_safe_scan_runs(
                    fixed=float(y), low=xa, high=xb, vertical=False,
                    minimum_m=4.0,
                ))

    ordered = []
    while pending:
        index, reverse = min(
            ((i, math.dist(current, end) < math.dist(current, start))
             for i, (start, end) in enumerate(pending)),
            key=lambda item: min(math.dist(current, pending[item[0]][0]),
                                 math.dist(current, pending[item[0]][1])),
        )
        start, end = pending.pop(index)
        if reverse:
            start, end = end, start
        ordered.append((start, end))
        current = end
    return ordered


def candidate_blocks(
    coarse: CoarseMap, selected: np.ndarray,
    records: list[dict[str, Any]],
) -> list[tuple[int, int, int, int, int]]:
    """Bound each detailed 0.2 m map to a small part of the large AOI."""
    hot = np.zeros(selected.shape, dtype=bool)
    uncertain = np.zeros(selected.shape, dtype=bool)
    for record in records:
        index = record["row"], record["col"]
        hot[index] = record["label"] == "hot"
        uncertain[index] = record["label"] == "uncertain"
    blocks = []
    for row in range(0, coarse.rows, FINE_CLUSTER_TILES):
        row_end = min(coarse.rows, row + FINE_CLUSTER_TILES)
        for col in range(0, coarse.cols, FINE_CLUSTER_TILES):
            col_end = min(coarse.cols, col + FINE_CLUSTER_TILES)
            if np.any(selected[row:row_end, col:col_end]):
                priority = (0 if np.any(hot[row:row_end, col:col_end])
                            else 1 if np.any(uncertain[row:row_end, col:col_end])
                            else 2)
                blocks.append((priority, row, row_end, col, col_end))
    return blocks


def hackathon_mission(ctx: Any) -> Iterator[Any]:
    """Scan all legal crop at 25 m, then inspect and treat candidate blocks."""
    log = ctx.world.log_info
    home = ctx.senses.pose.current_position
    home_north, home_east = home.x, home.y
    sprayer = ctx.services.sprayer
    save_stress_areas([])  # clear detections left by an earlier run
    coarse_legs = coarse_scan_segments()
    coarse_waypoints = [
        (x, y, COARSE_ALT_M) for start, end in coarse_legs for x, y in (start, end)
    ]
    detours = preflight_no_fly_routes((home_east, home_north), [coarse_waypoints])
    log(f"[AOI] whole crop polygon; {len(coarse_legs)} coarse legs at 25 m; "
        f"scan distance={sum(math.dist(a, b) for a, b in coarse_legs):.0f} m")
    log(f"[NFZ] coarse route preflight passed; {detours} detour(s)")

    coarse = CoarseMap()
    yield takeoff(alt_m=COARSE_ALT_M)
    for leg_index, (start, end) in enumerate(coarse_legs, start=1):
        yield from recharge_if_needed(ctx, resume_alt_m=COARSE_ALT_M)
        yield from safe_fly_to(
            ctx, east=start[0], north=start[1], alt_m=COARSE_ALT_M,
            target_speed=SURVEY_TRANSIT_SPEED_M_S,
            name=f"coarse_{leg_index:02d}_start",
        )
        mapping = ctx.scheduler.schedule(
            lambda: coarse.snap(ctx), hz=MAP_HZ,
            group=ScheduleGroup.MEDIA, name=f"coarse_mapper_{leg_index:02d}",
            now=ctx.world.now(),
        )
        try:
            scan_step, scan_result = timed_fly_to(
                ctx, east=end[0], north=end[1], alt_m=COARSE_ALT_M,
                target_speed=COARSE_SPEED_M_S, mode="coverage",
                name=f"coarse_{leg_index:02d}_scan",
            )
            yield scan_step
            if scan_result["timed_out"]:
                ctx.world.log_warn(
                    f"[AOI] coarse leg {leg_index} incomplete; "
                    "unobserved tiles will be checked at 5 m"
                )
        finally:
            ctx.scheduler.unschedule(mapping)

    selected, records = coarse.candidates()
    crop_total = int(np.sum(coarse.green + coarse.yellow + coarse.brown))
    seen_total = int(np.sum(coarse.seen))
    safe_tiles = int(np.count_nonzero(coarse.safe))
    observed_tiles = int(np.count_nonzero((coarse.seen >= 40) & coarse.safe))
    observed_fraction = observed_tiles / max(1, safe_tiles)
    reliable = (coarse.pictures >= 10 and observed_fraction >= 0.45
                and crop_total / max(1, seen_total) >= 0.12)
    if not reliable:
        ctx.world.log_warn(
            f"[AOI] coarse evidence unreliable (frames={coarse.pictures}, "
            f"observed={observed_fraction:.0%}); returning home instead of "
            "declaring the field healthy"
        )
    counts = {label: sum(record["label"] == label for record in records)
              for label in ("hot", "uncertain", "uniform_green",
                            "uniform_yellow", "cold", "noncrop", "excluded")}
    coarse_report = {
        "stage": "coarse_25m_screening_not_final_detection",
        "aoi_polygon_enu": AOI_POLYGON_ENU, "bounds_enu": aoi_bounds(),
        "tile_m": COARSE_TILE_M, "pictures": coarse.pictures,
        "observed_fraction": round(observed_fraction, 3),
        "reliable": reliable, "counts": counts, "tiles": records,
    }
    COARSE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    COARSE_MAP_PATH.write_text(json.dumps(coarse_report, separators=(",", ":")))
    log(f"[AOI] coarse pictures={coarse.pictures}; tiles={counts}; "
        f"selected={int(np.count_nonzero(selected))}; map={COARSE_MAP_PATH}")
    hot_centers = [record["center_enu"] for record in records
                   if record["label"] == "hot"]
    log(f"[AOI] mottled hot tile centres ENU (first 12): {hot_centers[:12]}")
    candidate_centers = [
        (record["label"], record["center_enu"])
        for record in records if record["label"] in ("hot", "uncertain")
    ]
    if reliable:
        for offset in range(0, len(candidate_centers), 20):
            log(f"[AOI] candidates {offset + 1}-"
                f"{min(offset + 20, len(candidate_centers))}: "
                f"{candidate_centers[offset:offset + 20]}")

    all_areas: list[dict[str, Any]] = []
    blocks = candidate_blocks(coarse, selected, records) if reliable else []
    log(f"[AOI] targeted 5 m scan blocks={len(blocks)}")
    block_index = 0
    while blocks:
        pose = ctx.senses.pose.current_position
        current = (float(pose.y), float(pose.x))
        closest = min(range(len(blocks)), key=lambda index: (
            blocks[index][0],
            math.dist(
                current,
                (coarse.x0 + 0.5 * (blocks[index][3] + blocks[index][4]) * COARSE_TILE_M,
                 coarse.y0 + 0.5 * (blocks[index][1] + blocks[index][2]) * COARSE_TILE_M),
            ),
        ))
        priority, row0, row1, col0, col1 = blocks.pop(closest)
        fine_legs = fine_scan_segments(
            coarse, selected, current,
            row_range=(row0, row1), col_range=(col0, col1),
        )
        if not fine_legs:
            log(f"[AOI] block ({row0},{col0}) has no legal fine-scan leg; skipped")
            continue
        block_index += 1
        fine_waypoints = [
            (x, y, FINE_ALT_M) for start, end in fine_legs for x, y in (start, end)
        ]
        detours = preflight_no_fly_routes(current, [fine_waypoints])
        log(f"[AOI] block {block_index}: priority={priority}, "
            f"tile_rows={row0}:{row1}, tile_cols={col0}:{col1}, "
            f"{len(fine_legs)} fine scan legs; NFZ detours={detours}")
        areas = yield from scan_detect_spray(
            ctx, survey_index=block_index, waypoints=fine_waypoints,
            sprayer=sprayer, scan_segments=fine_legs, prior_areas=all_areas,
        )
        all_areas.extend(areas)
        save_stress_areas(all_areas)
        log(f"[AOI] block {block_index} done; total stress areas={len(all_areas)}")

    yield from recharge_if_needed(ctx, resume_alt_m=CHARGE_TRANSIT_ALT_M)
    yield from safe_fly_to(
        ctx, north=home_north, east=home_east,
        alt_m=CHARGE_TRANSIT_ALT_M, target_speed=SURVEY_TRANSIT_SPEED_M_S,
        name="return_home",
    )
    yield brake(name="pre_land")
    yield land()


def scan_detect_spray(
    ctx: Any,
    *,
    survey_index: int,
    waypoints: list[tuple[float, float, float]],
    sprayer: Any,
    scan_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    prior_areas: list[dict[str, Any]] | None = None,
) -> Iterator[Any]:
    """Scan, detect and immediately spray one survey cluster."""
    tag = f"survey_{survey_index:02d}"
    log = ctx.world.log_info
    stress_map = StressMap(waypoints, scan_segments=scan_segments)

    # If the previous survey ended below 30%, charge before travelling to
    # this cluster. No mapping is active during the charging detour.
    yield from recharge_if_needed(ctx, resume_alt_m=waypoints[0][2])

    # Fly to this cluster's first waypoint without mapping the transit path.
    start_x, start_y, start_z = waypoints[0]
    yield from safe_fly_to(
        ctx,
        north=start_y,
        east=start_x,
        alt_m=start_z,
        target_speed=(SURVEY_TRANSIT_SPEED_M_S if survey_index > 1
                      else SURVEY_SPEED_M_S),
        name=f"{tag}_start",
    )

    # Map only while flying each actual survey leg. If charging is needed,
    # return to the last waypoint first and keep that transit out of the map.
    if scan_segments is None:
        legs = [(waypoints[i - 1][:2], waypoints[i][:2])
                for i in range(1, len(waypoints))]
    else:
        legs = scan_segments
    for leg_index, (start, end) in enumerate(legs, start=1):
        yield from recharge_if_needed(ctx, resume_alt_m=FINE_ALT_M)
        # Transit between selected tiles is never used for colour mapping.
        yield from safe_fly_to(
            ctx, east=start[0], north=start[1], alt_m=FINE_ALT_M,
            target_speed=SURVEY_TRANSIT_SPEED_M_S,
            name=f"{tag}_leg_{leg_index:02d}_start",
        )
        if not _segment_clear(start, end):
            raise RuntimeError(f"Unsafe fine scan segment: {start} -> {end}")
        mapping = ctx.scheduler.schedule(
            lambda: stress_map.snap(ctx), hz=MAP_HZ,
            group=ScheduleGroup.MEDIA,
            name=f"{tag}_mapper_leg_{leg_index:02d}",
            now=ctx.world.now(),
        )
        try:
            scan_step, scan_result = timed_fly_to(
                ctx, east=end[0], north=end[1], alt_m=FINE_ALT_M,
                target_speed=SURVEY_SPEED_M_S, mode="coverage",
                name=f"{tag}_leg_{leg_index:02d}_of_{len(legs)}",
            )
            yield scan_step
            if scan_result["timed_out"]:
                log(f"[AOI] fine scan leg {leg_index} incomplete")
        finally:
            ctx.scheduler.unschedule(mapping)

    # Detect only from frames collected in this survey.
    areas = stress_map.stress_areas()
    save_stress_areas((prior_areas or []) + areas)
    log(
        f"[EXAMPLE] {tag}: {stress_map.pictures} picture(s) -> "
        f"{len(areas)} stress area(s); colour stats={stress_map.stats}; "
        "spraying now"
    )

    # Build all safe lanes first. Contour order is arbitrary and a fixed
    # polygon-by-polygon sweep can make the drone cross the field with the
    # valve closed before treating a neighbouring patch.
    pending_lanes = []
    for area_index, area in enumerate(areas, start=1):
        lanes = stress_map.safe_spray_lanes(area["polygon"])
        if not lanes:
            log(
                f"[SPRAY] {tag}: area {area_index} has no safe lane "
                "wide enough for the spray cone; skipped"
            )
            continue
        log(f"[SPRAY] {tag}: area {area_index} -> {len(lanes)} safe lane(s)")
        for lane_index, (start, end) in enumerate(lanes, start=1):
            if not _segment_clear(start, end, NO_FLY_SPRAY_CLEARANCE_M):
                log(
                    f"[NFZ] {tag}: area {area_index} lane {lane_index} "
                    "touches a no-fly zone; skipped"
                )
                continue
            pending_lanes.append((area_index, lane_index, start, end))

    last_endpoint = (waypoints[-1][0], waypoints[-1][1])
    previous_area = None
    while pending_lanes:
        # Recharge first, because the nearest lane can change after returning
        # from a charging station. The valve is closed during this transit.
        yield from recharge_if_needed(ctx, resume_alt_m=SPRAY_ALT_M)
        pose = ctx.senses.pose.current_position
        current = (pose.y, pose.x) if pose is not None else last_endpoint
        area_index, lane_index, start, end = pop_nearest_spray_lane(
            pending_lanes, current,
        )
        if area_index != previous_area:
            log(
                f"[SPRAY] {tag}: next area {area_index}, "
                f"approach {math.dist(current, start):.1f} m"
            )
            previous_area = area_index
        lane_name = f"{tag}_area_{area_index:02d}_lane_{lane_index:02d}"
        approach_route = safe_route_from_pose(ctx, start[0], start[1])
        if len(approach_route) > 1:
            log(f"[NFZ] {lane_name}: routing around no-fly zone before spraying")
        approach_failed = False
        for detour_index, (east, north) in enumerate(approach_route, start=1):
            approach_step, approach_result = timed_fly_to(
                ctx, north=north, east=east, alt_m=SPRAY_ALT_M,
                target_speed=SPRAY_SPEED_M_S, mode="coverage",
                name=(f"{lane_name}_start" if detour_index == len(approach_route)
                      else f"{lane_name}_nfz_detour_{detour_index:02d}"),
            )
            yield approach_step
            if approach_result["timed_out"]:
                approach_failed = True
                break
        if approach_failed:
            log(
                f"[EXAMPLE] {tag}: skipping area {area_index}, "
                f"lane {lane_index}; start is unreachable"
            )
            continue
        last_endpoint = start

        opened = yield from ensure_sprayer_open(
            ctx,
            sprayer,
            name=lane_name,
        )
        if not opened:
            # Never fly a scoring lane while the valve is known closed.
            if sprayer.state:
                close_step, close_result = wait_for_sprayer_condition(
                    ctx,
                    condition=lambda: not sprayer.state,
                    name=f"{lane_name}_wait_closed_after_failed_open",
                )
                yield close_step
                if close_result["timed_out"] or sprayer.state:
                    raise RuntimeError(
                        f"{lane_name}: sprayer did not close; stopping before transit"
                    )
            continue

        try:
            spray_step, spray_result = timed_fly_to(
                ctx,
                north=end[1],
                east=end[0],
                alt_m=SPRAY_ALT_M,
                target_speed=SPRAY_SPEED_M_S,
                mode="coverage",
                name=(
                    f"{tag}_area_{area_index:02d}_"
                    f"lane_{lane_index:02d}_spray"
                ),
            )
            yield spray_step
            if spray_result["timed_out"]:
                log(
                    f"[EXAMPLE] {tag}: area {area_index}, lane "
                    f"{lane_index} ended early due to unreachable target"
                )
        finally:
            # Always close the valve, including cancellation/error paths.
            accepted = sprayer.off()
            log(
                f"[SPRAY] {lane_name}: OFF accepted={accepted}, "
                f"state={sprayer.state}"
            )
        if sprayer.state:
            close_step, close_result = wait_for_sprayer_condition(
                ctx,
                condition=lambda: not sprayer.state,
                name=f"{lane_name}_wait_closed",
            )
            yield close_step
            if close_result["timed_out"] or sprayer.state:
                raise RuntimeError(
                    f"{lane_name}: sprayer did not close; stopping before transit"
                )
        last_endpoint = end

    log(f"[EXAMPLE] {tag}: spraying done")
    return areas


def pop_nearest_spray_lane(
    pending: list[
        tuple[int, int, tuple[float, float], tuple[float, float]]
    ],
    current: tuple[float, float],
) -> tuple[int, int, tuple[float, float], tuple[float, float]]:
    """Choose the closest remaining lane endpoint; fly the lane either way."""
    if not pending:
        raise ValueError("No spray lanes remain")

    best_index = 0
    best_reverse = False
    best_distance_sq = math.inf
    for index, (_, _, start, end) in enumerate(pending):
        start_distance_sq = (
            (current[0] - start[0]) ** 2 + (current[1] - start[1]) ** 2
        )
        end_distance_sq = (
            (current[0] - end[0]) ** 2 + (current[1] - end[1]) ** 2
        )
        if min(start_distance_sq, end_distance_sq) < best_distance_sq:
            best_index = index
            best_reverse = end_distance_sq < start_distance_sq
            best_distance_sq = min(start_distance_sq, end_distance_sq)

    area_index, lane_index, start, end = pending.pop(best_index)
    if best_reverse:
        start, end = end, start
    return area_index, lane_index, start, end


hackathon_mission.requires_senses = [
    "pose", "obstacle", "status", "camera", "battery"
]


def save_stress_areas(areas: list[dict[str, Any]]) -> None:
    """Write ``{"stress_area": areas}`` whole, so a copy never catches half a file.

    The temp file must not be named ``stress_area.json``: the app picks up
    that name anywhere under captures.
    """
    STRESS_AREA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STRESS_AREA_PATH.with_name(".stress_area.json.tmp")
    tmp.write_text(json.dumps({"stress_area": areas}, separators=(",", ":")))
    tmp.replace(STRESS_AREA_PATH)


class StressMap:
    """Map rice colours and detect yellow patches in green or brown rice."""

    def __init__(
        self,
        waypoints_enu: list[tuple[float, float, float]],
        scan_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> None:
        margin = 25.0                   # room around the route for the picture footprint
        xs = [p[0] for p in waypoints_enu]
        ys = [p[1] for p in waypoints_enu]
        self.x0 = min(xs) - margin
        self.y0 = min(ys) - margin
        rows = int((max(ys) + margin - self.y0) / CELL_M) + 1
        cols = int((max(xs) + margin - self.x0) / CELL_M) + 1
        self.seen = np.zeros((rows, cols), dtype=np.int32)
        self.green = np.zeros((rows, cols), dtype=np.int32)
        self.yellow = np.zeros((rows, cols), dtype=np.int32)
        self.brown = np.zeros((rows, cols), dtype=np.int32)
        self.dark = np.zeros((rows, cols), dtype=np.int32)
        self.spray_mask = np.zeros((rows, cols), dtype=np.uint8)
        self.pictures = 0
        self._last_seq = -1
        self.exposure_gain_sum = 0.0
        self.exposure_frames = 0
        self.stats: dict[str, float | int] = {}
        self.aoi_mask: np.ndarray | None = None

        # A camera frame also sees roads and neighbouring parcels. Limit
        # candidate spray areas to the 5 m corridor around this survey route.
        self.survey_mask = np.zeros((rows, cols), dtype=np.uint8)
        def pixel(point: tuple[float, float]) -> tuple[int, int]:
            x, y = point
            return (int(round((x - self.x0) / CELL_M)),
                    int(round((y - self.y0) / CELL_M)))

        thickness = max(1, int(round(2 * SURVEY_CORRIDOR_RADIUS_M / CELL_M)))
        if scan_segments is not None:
            for start, end in scan_segments:
                cv2.line(self.survey_mask, pixel(start), pixel(end), 1, thickness)
            # Clip mapped crop to the true AOI polygon, not its bounding box.
            self.aoi_mask = np.zeros((rows, cols), dtype=np.uint8)
            aoi_vertices = np.array([pixel(p) for p in AOI_POLYGON_ENU], dtype=np.int32)
            cv2.fillPoly(self.aoi_mask, [aoi_vertices], 1)
            forbidden = np.zeros((rows, cols), dtype=np.uint8)
            for polygon in NO_FLY_ZONES_ENU.values():
                px = [p[0] for p in polygon]
                py = [p[1] for p in polygon]
                if (max(px) < self.x0 - NO_FLY_CLEARANCE_M
                        or min(px) > self.x0 + cols * CELL_M + NO_FLY_CLEARANCE_M
                        or max(py) < self.y0 - NO_FLY_CLEARANCE_M
                        or min(py) > self.y0 + rows * CELL_M + NO_FLY_CLEARANCE_M):
                    continue
                vertices = np.array([pixel(p) for p in polygon], dtype=np.int32)
                cv2.fillPoly(forbidden, [vertices], 1)
            if np.any(forbidden):
                buffer_cells = math.ceil(NO_FLY_CLEARANCE_M / CELL_M)
                buffer_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * buffer_cells + 1, 2 * buffer_cells + 1),
                )
                forbidden = cv2.dilate(forbidden, buffer_kernel)
                self.aoi_mask[forbidden > 0] = 0
            self.survey_mask &= self.aoi_mask
        else:
            route = [pixel((x, y)) for x, y, _ in waypoints_enu]
            if len(route) == 1:
                cv2.circle(self.survey_mask, route[0], thickness // 2, 1, -1)
            else:
                for start, end in zip(route, route[1:]):
                    cv2.line(self.survey_mask, start, end, 1, thickness)

    def snap(self, ctx: Any) -> None:
        """Read the latest RGB frame without writing a PNG to cloud storage."""
        camera = ctx.senses.camera
        if not camera.has_frame or camera.seq == self._last_seq:   # no new frame yet
            return
        pose = ctx.senses.pose.current_position
        if pose is None:
            return
        rgb = camera.decode()  # SkyTrack /camera is rgb8.
        if rgb is None:
            return
        self.add_picture(rgb, pose)
        self._last_seq = camera.seq
        self.pictures += 1

    def add_picture(self, rgb: np.ndarray, pose: Any) -> None:
        """Project every 2nd pixel and vote for its rice colour."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # SkyTrack's camera frames can be much darker than the published
        # calibration crops. Use the upper image (clear of the drone body)
        # to put its brightness onto a comparable scale before HSV voting.
        h, w = hsv.shape[:2]
        reference = hsv[:max(1, int(0.47 * h))]
        reference_pixels = reference[:, :, 2][
            (reference[:, :, 1] >= 40) & (reference[:, :, 2] >= 50)
        ]
        gain = 1.0
        if reference_pixels.size >= 100:
            reference_p75 = float(np.percentile(reference_pixels, 75))
            gain = float(np.clip(
                REFERENCE_VALUE_P75 / max(1.0, reference_p75),
                MIN_EXPOSURE_GAIN, MAX_EXPOSURE_GAIN,
            ))
            hsv[:, :, 2] = np.clip(
                hsv[:, :, 2].astype(np.float32) * gain, 0, 255,
            ).astype(np.uint8)
        self.exposure_gain_sum += gain
        self.exposure_frames += 1
        is_green = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH) > 0
        is_brown = cv2.inRange(hsv, BROWN_HSV_LOW, BROWN_HSV_HIGH) > 0
        is_dark = hsv[:, :, 2] < DARK_VALUE_MAX
        is_yellow = (
            (cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH) > 0)
            & ~is_brown
        )

        # Nadir camera: picture top = drone front.
        h, w = is_yellow.shape
        v, u = np.mgrid[0:h:2, 0:w:2]
        # The two nozzles and the white drone nose are fixed in the lower
        # part of SkyTrack's camera view. They are not field observations.
        occluded = (
            (v >= 0.47 * h) & (
                ((u >= 0.15 * w) & (u <= 0.34 * w))
                | ((u >= 0.67 * w) & (u <= 0.85 * w))
            )
        ) | (
            (v >= 0.75 * h) & (u >= 0.34 * w) & (u <= 0.67 * w)
        )
        focal = w / (2.0 * math.tan(CAMERA_HFOV_RAD / 2.0))
        height = -pose.z + CAMERA_HEIGHT_AT_HOME_M
        right = (u - w / 2.0) / focal * height          # metres right of the drone
        back = (v - h / 2.0) / focal * height           # metres behind the drone
        cos_h, sin_h = math.cos(pose.heading), math.sin(pose.heading)
        east = pose.y + right * cos_h - back * sin_h
        north = pose.x - right * sin_h - back * cos_h

        row = ((north - self.y0) / CELL_M).astype(int)
        col = ((east - self.x0) / CELL_M).astype(int)
        inside = (
            (row >= 0) & (row < self.seen.shape[0])
            & (col >= 0) & (col < self.seen.shape[1])
            & ~occluded
        )
        row, col = row[inside], col[inside]
        np.add.at(self.seen, (row, col), 1)
        np.add.at(self.green, (row, col), is_green[v, u][inside].astype(np.int32))
        np.add.at(self.yellow, (row, col), is_yellow[v, u][inside].astype(np.int32))
        np.add.at(self.brown, (row, col), is_brown[v, u][inside].astype(np.int32))
        np.add.at(self.dark, (row, col), is_dark[v, u][inside].astype(np.int32))

    def stress_areas(self) -> list[dict[str, Any]]:
        """Accept mottled yellow or light-brown patches within green rice."""
        observed = (self.seen >= MIN_SAMPLES_PER_CELL) & (self.survey_mask > 0)
        dark = observed & (self.dark >= np.ceil(self.seen * DARK_CELL_RATIO))
        dark_radius = max(1, round(DARK_BUFFER_M / CELL_M))
        dark_kernel = np.ones((2 * dark_radius + 1, 2 * dark_radius + 1), dtype=np.uint8)
        dark_buffer = cv2.dilate(dark.astype(np.uint8), dark_kernel) > 0
        eligible = observed & ~dark_buffer
        green = eligible & (
            self.green >= np.ceil(self.seen * MIN_GREEN_COLOR_RATIO)
        )
        stress_threshold = np.ceil(self.seen * MIN_STRESS_COLOR_RATIO)
        yellow = eligible & (self.yellow >= stress_threshold)
        brown = eligible & (self.brown >= stress_threshold)

        crop = green | yellow | brown
        crop_cells = int(np.count_nonzero(crop))
        green_cells = int(np.count_nonzero(green))
        green_fraction = green_cells / crop_cells if crop_cells else 0.0
        yellow_cells = int(np.count_nonzero(yellow))
        yellow_fraction = yellow_cells / crop_cells if crop_cells else 0.0

        def green_context(roi: np.ndarray, x: int, y: int) -> tuple[float, int, int]:
            """Green inside a patch, and on the four sides around its bounds."""
            height, width = roi.shape
            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            filled = np.zeros_like(roi)
            if contours:
                cv2.drawContours(filled, contours, -1, 1, -1)
            # Ignore the outer 0.4 m: a solid yellow rectangle otherwise
            # appears to contain green due to rasterisation at its border.
            inner = cv2.erode(
                filled, np.ones((5, 5), dtype=np.uint8),
                borderType=cv2.BORDER_CONSTANT, borderValue=0,
            ) > 0
            observed = inner & eligible[y:y + height, x:x + width]
            green_inside = int(np.count_nonzero(
                observed & green[y:y + height, x:x + width]
            ))
            observed_count = int(np.count_nonzero(observed))
            interior_fraction = (
                green_inside / observed_count if observed_count else 0.0
            )

            margin = max(1, round(GREEN_CONTEXT_MARGIN_M / CELL_M))
            sides = (
                (slice(max(0, y - margin), y), slice(x, x + width)),
                (slice(y + height, min(green.shape[0], y + height + margin)),
                 slice(x, x + width)),
                (slice(y, y + height), slice(max(0, x - margin), x)),
                (slice(y, y + height),
                 slice(x + width, min(green.shape[1], x + width + margin))),
            )
            green_sides = 0
            for side in sides:
                side_observed = int(np.count_nonzero(eligible[side]))
                if side_observed and (
                    np.count_nonzero(green[side]) / side_observed
                    >= MIN_SIDE_GREEN_FRACTION
                ):
                    green_sides += 1
            return interior_fraction, green_inside, green_sides

        # A yellow survey with almost no healthy green reference is a
        # uniformly coloured field, not evidence of localised stress.
        accepted_yellow = np.zeros_like(yellow, dtype=np.uint8)
        yellow_components = 0
        if (
            green_fraction >= MIN_GREEN_FRACTION
            and yellow_fraction < MAX_BASELINE_YELLOW_FRACTION
        ):
            group_radius = max(1, round(MOTTLE_GROUP_GAP_M / (2 * CELL_M)))
            group_kernel = np.ones(
                (2 * group_radius + 1, 2 * group_radius + 1), dtype=np.uint8,
            )
            yellow_clean = cv2.morphologyEx(
                yellow.astype(np.uint8), cv2.MORPH_CLOSE, group_kernel,
            )
            yellow_clean &= eligible.astype(np.uint8)
            distance_to_green = cv2.distanceTransform(
                (~green).astype(np.uint8), cv2.DIST_L2, 3,
            )
            near_green = distance_to_green <= GREEN_NEIGHBOR_RADIUS_M / CELL_M
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                yellow_clean, connectivity=8,
            )
            for label in range(1, count):
                if stats[label, cv2.CC_STAT_AREA] * CELL_M ** 2 < MIN_AREA_M2:
                    continue
                x, y, width, height, _ = stats[label]
                roi = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
                interior_green, green_inside, _ = green_context(roi, x, y)
                # A small, solid-yellow parcel is normal even when green
                # surrounds it. Yellow needs an actual mottled interior.
                if (
                    interior_green >= MIN_INTERIOR_GREEN_FRACTION
                    and green_inside >= MIN_INTERIOR_GREEN_CELLS
                    and np.any((roi > 0) & near_green[y:y + height, x:x + width])
                ):
                    target = accepted_yellow[y:y + height, x:x + width]
                    target[roi > 0] = 1
                    yellow_components += 1

        # Light-brown crop is stressed only if embedded in green rice.
        # Brown roads/ditches generally have green on at most two sides.
        accepted_brown = np.zeros_like(brown, dtype=np.uint8)
        brown_components = 0
        brown_clean = cv2.morphologyEx(
            brown.astype(np.uint8), cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        brown_clean &= eligible.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            brown_clean, connectivity=8,
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] * CELL_M ** 2 < MIN_AREA_M2:
                continue
            x, y, width, height, _ = stats[label]
            roi = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
            interior_green, green_inside, green_sides = green_context(roi, x, y)
            if (
                green_sides >= MIN_GREEN_CONTEXT_SIDES
                or (
                    interior_green >= MIN_INTERIOR_GREEN_FRACTION
                    and green_inside >= MIN_INTERIOR_GREEN_CELLS
                )
            ):
                target = accepted_brown[y:y + height, x:x + width]
                target[roi > 0] = 1
                brown_components += 1

        stressed = (accepted_yellow > 0) | (accepted_brown > 0)
        stressed = cv2.morphologyEx(
            stressed.astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        stressed &= eligible.astype(np.uint8)

        # The outer contour below may enclose healthy green holes. A nozzle
        # centre must itself be on dominant yellow/brown, and the footprint
        # beneath it must mostly cover stress rather than healthy crop.
        dominant_stress = eligible & (
            ((accepted_yellow > 0) & (self.yellow > self.green))
            | ((accepted_brown > 0) & (self.brown > self.green))
        )
        spray_radius = max(1, math.ceil(
            (SPRAY_SWATH_M / 2.0 + SPRAY_FOOTPRINT_MARGIN_M) / CELL_M
        ))
        spray_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * spray_radius + 1, 2 * spray_radius + 1),
        )
        footprint = spray_kernel.astype(np.float32)
        stress_cover = cv2.filter2D(
            dominant_stress.astype(np.float32), -1, footprint,
        )
        healthy_only = green & ~dominant_stress
        healthy_cover = cv2.filter2D(
            healthy_only.astype(np.float32), -1, footprint,
        )
        visible_cover = cv2.filter2D(
            eligible.astype(np.float32), -1, footprint,
        )
        dark_nearby = cv2.dilate(dark.astype(np.uint8), spray_kernel) > 0
        self.spray_mask = (
            dominant_stress
            & (stress_cover >= MIN_LOCAL_STRESS_FRACTION
               * (stress_cover + healthy_cover))
            & (visible_cover >= MIN_VISIBLE_FOOTPRINT_FRACTION
               * np.count_nonzero(spray_kernel))
            & ~dark_nearby
        ).astype(np.uint8)
        if self.aoi_mask is not None:
            # The entire nozzle footprint must stay inside crop and outside
            # the residential buffer, not just the centre of the drone.
            self.spray_mask &= cv2.erode(self.aoi_mask, spray_kernel)
        contours, _ = cv2.findContours(
            stressed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        areas = []
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) * CELL_M ** 2 < MIN_AREA_M2:
                continue
            _x, _y, width, height = cv2.boundingRect(contour)
            if (
                min(width, height) * CELL_M <= MAX_LINE_ARTIFACT_WIDTH_M
                and max(width, height) * CELL_M >= MIN_LINE_ARTIFACT_LENGTH_M
            ):
                continue
            polygon = [[float(self.x0 + (col + 0.5) * CELL_M), float(self.y0 + (row + 0.5) * CELL_M)]
                       for col, row in contour[:, 0]]
            areas.append({"class": "stressed", "polygon": polygon})
        self.stats = {
            "crop_cells": crop_cells,
            "green_cells": green_cells,
            "yellow_cells": yellow_cells,
            "brown_cells": int(np.count_nonzero(brown)),
            "dark_cells": int(np.count_nonzero(dark)),
            "mean_exposure_gain": round(
                self.exposure_gain_sum / max(1, self.exposure_frames), 2,
            ),
            "green_fraction": round(green_fraction, 3),
            "yellow_fraction": round(yellow_fraction, 3),
            "accepted_yellow_components": yellow_components,
            "accepted_brown_components": brown_components,
            "dominant_stress_cells": int(np.count_nonzero(dominant_stress)),
            "safe_spray_cells": int(np.count_nonzero(self.spray_mask)),
            "areas": len(areas),
        }
        return areas

    def safe_spray_lanes(
        self, polygon: list[list[float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Split polygon lanes wherever the nozzle would cross healthy crop."""
        safe_lanes = []
        for start, end in spray_lanes(polygon):
            length = math.dist(start, end)
            samples = max(2, math.ceil(length / (CELL_M / 2.0)) + 1)
            t = np.linspace(0.0, 1.0, samples)
            east = start[0] + (end[0] - start[0]) * t
            north = start[1] + (end[1] - start[1]) * t
            col = np.floor((east - self.x0) / CELL_M).astype(int)
            row = np.floor((north - self.y0) / CELL_M).astype(int)
            inside = (
                (row >= 0) & (row < self.spray_mask.shape[0])
                & (col >= 0) & (col < self.spray_mask.shape[1])
            )
            allowed = np.zeros(samples, dtype=bool)
            allowed[inside] = self.spray_mask[row[inside], col[inside]] > 0
            switches = np.flatnonzero(np.diff(np.r_[False, allowed, False]))
            for first, stop in zip(switches[::2], switches[1::2]):
                if (stop - first - 1) * length / (samples - 1) < MIN_SPRAY_RUN_M:
                    continue
                safe_lanes.append((
                    (float(east[first]), float(north[first])),
                    (float(east[stop - 1]), float(north[stop - 1])),
                ))
        return safe_lanes


def spray_lanes(polygon: list[list[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Parallel spray lanes over the polygon, clipped to it, as (start, end) ENU pairs.

    Lanes run along the polygon's longer side and alternate direction.
    """
    along_north = (max(p[1] for p in polygon) - min(p[1] for p in polygon)) > \
                  (max(p[0] for p in polygon) - min(p[0] for p in polygon))
    # Work in (along, across) coordinates, then convert back to ENU.
    points = [(p[1], p[0]) if along_north else (p[0], p[1]) for p in polygon]
    out = (lambda along, across: (across, along)) if along_north else \
          (lambda along, across: (along, across))

    low = min(p[1] for p in points)
    high = max(p[1] for p in points)
    count = max(1, math.ceil((high - low) / SPRAY_LANE_SPACING_M))
    step = (high - low) / count         # even spacing across the polygon

    lanes = []
    for i in range(count):
        across = low + (i + 0.5) * step
        runs = [(a, b) for a, b in _inside_runs(points, across) if b - a >= MIN_SPRAY_RUN_M]
        if i % 2:                       # reverse every other lane
            runs = [(b, a) for a, b in reversed(runs)]
        for a, b in runs:
            inset = _inset_spray_lane(out(a, across), out(b, across))
            if inset is not None:
                lanes.append(inset)
    return lanes


def _inset_spray_lane(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Move both endpoints inward so the planner does not target a boundary."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < MIN_SPRAY_RUN_M:
        return None

    # Keep short lanes usable by reducing the inset, but always leave a
    # non-zero segment after trimming.
    inset_m = min(
        SPRAY_ENDPOINT_INSET_M,
        max(0.0, (length - MIN_SPRAY_RUN_M) / 2.0),
    )
    ux, uy = dx / length, dy / length
    return (
        (start[0] + ux * inset_m, start[1] + uy * inset_m),
        (end[0] - ux * inset_m, end[1] - uy * inset_m),
    )


def _inside_runs(points: list[tuple[float, float]], across: float) -> list[tuple[float, float]]:
    """(start, end) stretches where the line at ``across`` is inside the polygon."""
    hits = []
    for (a_along, a_across), (b_along, b_across) in zip(points, points[1:] + points[:1]):
        if (a_across > across) == (b_across > across):
            continue                    # edge does not cross the line
        t = (across - a_across) / (b_across - a_across)
        hits.append(a_along + t * (b_along - a_along))
    hits.sort()
    return list(zip(hits[::2], hits[1::2]))


def main() -> None:
    with boot_drone() as drone:         # the app provides the RGB camera sense
        drone.add_service(BatteryMonitor())
        drone.add_service(Sprayer())
        drone.fly(hackathon_mission)
        drone.run()


if __name__ == "__main__":
    main()
