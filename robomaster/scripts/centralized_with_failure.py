# -*-coding:utf-8-*-
"""Three-partition coverage with manual robot-2 failure recovery.

Press F to stop robot_2. The first of robot_1 or robot_3 to finish its own
partition plans a radius-safe route to robot_2's unfinished lawnmower path and
completes the remaining reachable waypoints. Only JSON metadata is recorded.
"""

import argparse
import copy
from datetime import datetime
import heapq
import json
import math
import msvcrt
from pathlib import Path
import sys
import threading
import time
import types


SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

# Robot.initialize() creates a camera object even when no stream is requested.
sys.modules["libmedia_codec"] = types.ModuleType("libmedia_codec")
from robomaster import media, robot


class NoMediaLiveView:
    """Satisfy the SDK camera module without constructing media decoders."""

    def __init__(self, ep_robot):
        self._robot = ep_robot

    def start_video_stream(self, display=False, addr=None, ip_proto="tcp"):
        return True

    def stop_video_stream(self):
        return True

    def stop(self):
        pass


AREA_X_METERS = 4.1
AREA_Y_METERS = 1.8
ROBOT_COUNT = 3
PARTITION_WIDTH_METERS = AREA_X_METERS / ROBOT_COUNT
ROBOT_SPECS = (
    ("robot_1", "3JKCH8800100VW", 0.0),
    ("robot_2", "3JKCH8800100RC", PARTITION_WIDTH_METERS),
    ("robot_3", "3JKCH8800100VZ", 2.0 * PARTITION_WIDTH_METERS),
)
FAILED_ROBOT_NAME = "robot_2"
TAKEOVER_CANDIDATES = ("robot_1", "robot_3")

ROBOT_RADIUS_METERS = 10.0 * 0.0254
ROBOT_DIAMETER_METERS = 2.0 * ROBOT_RADIUS_METERS
COLLISION_BUFFER_METERS = 0.10
ROBOT_CLEARANCE_METERS = ROBOT_DIAMETER_METERS + COLLISION_BUFFER_METERS
EDGE_BUFFER_METERS = 0.01
EDGE_MARGIN_METERS = ROBOT_RADIUS_METERS + EDGE_BUFFER_METERS
PARTITION_EDGE_MARGIN_METERS = ROBOT_CLEARANCE_METERS / 2.0
BOUNDARY_ODOMETRY_TOLERANCE_METERS = 0.03
BOUNDARY_RECOVERY_TOLERANCE_METERS = 0.10
PLANNER_RING_BUFFER_METERS = 0.01
PLANNER_RING_POINT_COUNT = 16
FAILED_ROBOT_PLANNER_BUFFER_METERS = 0.12
ACTIVE_ROBOT_PLANNER_BUFFER_METERS = 0.03
FAILED_ROBOT_PLANNING_CLEARANCE_METERS = (
    ROBOT_CLEARANCE_METERS + FAILED_ROBOT_PLANNER_BUFFER_METERS
)
ACTIVE_ROBOT_PLANNING_CLEARANCE_METERS = (
    ROBOT_CLEARANCE_METERS + ACTIVE_ROBOT_PLANNER_BUFFER_METERS
)
PEER_WAIT_TIMEOUT_SECONDS = 45.0

DEFAULT_LINEAR_SPEED_MPS = 0.08
DEFAULT_TURN_SPEED_DPS = 30.0
METADATA_SAMPLE_PERIOD_SECONDS = 0.25
CONTROL_PERIOD_SECONDS = 0.05
DRIVE_COMMAND_TIMEOUT_SECONDS = 0.3
MOTION_SETTLE_SECONDS = 0.4
TURN_TOLERANCE_DEGREES = 2.0
HEADING_DRIFT_TOLERANCE_DEGREES = 4.0
DISTANCE_TOLERANCE_METERS = 0.025
MIN_TURN_SPEED_DPS = 5.0
MIN_LINEAR_SPEED_MPS = 0.08
MOTION_TIMEOUT_MULTIPLIER = 60.0
MOTION_TIMEOUT_MARGIN_SECONDS = 200.0
MIN_DISTANCE_BETWEEN_HEADING_CORRECTIONS_METERS = 0.35


class ManualRobotFailure(Exception):
    """Stop one robot without stopping the experiment."""


def utc_timestamp():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def round_point(point):
    return {"x": round(point[0], 4), "y": round(point[1], 4)}


def normalize_angle_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def point_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def point_inside_arena(point, tolerance=0.0):
    return (
        EDGE_MARGIN_METERS - tolerance
        <= point[0]
        <= AREA_X_METERS - EDGE_MARGIN_METERS + tolerance
        and EDGE_MARGIN_METERS - tolerance
        <= point[1]
        <= AREA_Y_METERS - EDGE_MARGIN_METERS + tolerance
    )


def arena_boundary_violation(point):
    """Return distance outside the permitted robot-center rectangle."""
    x_min = EDGE_MARGIN_METERS
    x_max = AREA_X_METERS - EDGE_MARGIN_METERS
    y_min = EDGE_MARGIN_METERS
    y_max = AREA_Y_METERS - EDGE_MARGIN_METERS
    x_error = max(x_min - point[0], 0.0, point[0] - x_max)
    y_error = max(y_min - point[1], 0.0, point[1] - y_max)
    return math.hypot(x_error, y_error)


def segment_respects_arena(start, end):
    """Allow a small odometry overrun only when the segment returns inward."""
    start_violation = arena_boundary_violation(start)
    end_violation = arena_boundary_violation(end)
    normal_tolerance = BOUNDARY_ODOMETRY_TOLERANCE_METERS

    if (
        start_violation <= normal_tolerance
        and end_violation <= normal_tolerance
    ):
        return True

    return (
        start_violation > normal_tolerance
        and start_violation <= BOUNDARY_RECOVERY_TOLERANCE_METERS
        and end_violation <= normal_tolerance
        and end_violation < start_violation
    )


def distance_point_to_segment(point, start, end):
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared <= 1e-12:
        return point_distance(point, start)
    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = (
        start[0] + projection * segment_x,
        start[1] + projection * segment_y,
    )
    return point_distance(point, closest)


def segment_is_clear(start, end, obstacles):
    if not segment_respects_arena(start, end):
        return False
    return all(
        distance_point_to_segment(center, start, end) >= clearance
        for center, clearance in obstacles
    )


def plan_clear_path(start, goal, obstacles):
    """Use a small visibility graph to route around circular obstacles."""
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    if segment_is_clear(start, goal, obstacles):
        return [goal]

    nodes = [start, goal]
    for center, clearance in obstacles:
        ring_radius = (
            clearance / math.cos(math.pi / PLANNER_RING_POINT_COUNT)
            + PLANNER_RING_BUFFER_METERS
        )
        for index in range(PLANNER_RING_POINT_COUNT):
            angle = 2.0 * math.pi * index / PLANNER_RING_POINT_COUNT
            candidate = (
                center[0] + ring_radius * math.cos(angle),
                center[1] + ring_radius * math.sin(angle),
            )
            if point_inside_arena(candidate):
                nodes.append(candidate)

    adjacency = [[] for _ in nodes]
    for first_index, first in enumerate(nodes):
        for second_index in range(first_index + 1, len(nodes)):
            second = nodes[second_index]
            if segment_is_clear(first, second, obstacles):
                distance = point_distance(first, second)
                adjacency[first_index].append((distance, second_index))
                adjacency[second_index].append((distance, first_index))

    distances = [float("inf")] * len(nodes)
    previous = [None] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, node_index = heapq.heappop(queue)
        if distance != distances[node_index]:
            continue
        if node_index == 1:
            break
        for edge_distance, neighbor_index in adjacency[node_index]:
            candidate = distance + edge_distance
            if candidate < distances[neighbor_index]:
                distances[neighbor_index] = candidate
                previous[neighbor_index] = node_index
                heapq.heappush(queue, (candidate, neighbor_index))

    if not math.isfinite(distances[1]):
        raise RuntimeError(
            "No radius-safe path from {0} to {1}".format(
                round_point(start),
                round_point(goal),
            )
        )

    indices = []
    node_index = 1
    while node_index is not None:
        indices.append(node_index)
        node_index = previous[node_index]
    indices.reverse()
    return [nodes[index] for index in indices[1:]]


def make_lane_positions():
    """Place center lanes so radius swaths cover the complete arena height."""
    minimum = EDGE_MARGIN_METERS
    maximum = AREA_Y_METERS - EDGE_MARGIN_METERS
    span = maximum - minimum
    interval_count = max(1, int(math.ceil(span / ROBOT_DIAMETER_METERS)))
    spacing = span / interval_count
    return [minimum + index * spacing for index in range(interval_count + 1)]


def build_robot_plan(name, partition_x_min):
    """Build centralized.py-style turns and forward-only translations."""
    partition_x_max = partition_x_min + PARTITION_WIDTH_METERS
    left_margin = (
        EDGE_MARGIN_METERS
        if math.isclose(partition_x_min, 0.0)
        else PARTITION_EDGE_MARGIN_METERS
    )
    right_margin = (
        EDGE_MARGIN_METERS
        if math.isclose(partition_x_max, AREA_X_METERS)
        else PARTITION_EDGE_MARGIN_METERS
    )
    x_min = partition_x_min + left_margin
    x_max = partition_x_max - right_margin
    if x_min >= x_max:
        raise RuntimeError("Partition is too narrow for the robot radius")

    lanes = make_lane_positions()
    waypoints = []
    actions = []
    world_x = x_min
    world_y = lanes[0]
    heading = 0.0

    for row_index, lane_y in enumerate(lanes):
        destination_x = x_max if row_index % 2 == 0 else x_min
        waypoint_index = len(waypoints)
        waypoints.append(
            {
                "kind": "traverse",
                "row": row_index,
                "world_end": round_point((destination_x, lane_y)),
            }
        )
        actions.append(
            {
                "kind": "traverse",
                "row": row_index,
                "waypoint_index": waypoint_index,
                "distance_m": round(abs(destination_x - world_x), 4),
                "heading_start_deg": heading,
                "heading_end_deg": heading,
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((destination_x, lane_y)),
            }
        )
        world_x = destination_x
        world_y = lane_y

        if row_index == len(lanes) - 1:
            continue

        next_lane_y = lanes[row_index + 1]
        turn_degrees = 90.0 if row_index % 2 == 0 else -90.0
        lane_waypoint_index = len(waypoints)
        waypoints.append(
            {
                "kind": "lane_shift",
                "row": row_index,
                "world_end": round_point((world_x, next_lane_y)),
            }
        )
        actions.append(
            {
                "kind": "turn_toward_next_lane",
                "row": row_index,
                "waypoint_index": lane_waypoint_index,
                "turn_deg": turn_degrees,
                "distance_m": 0.0,
                "heading_start_deg": heading,
                "heading_end_deg": heading + turn_degrees,
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((world_x, world_y)),
            }
        )
        heading += turn_degrees
        actions.append(
            {
                "kind": "lane_shift",
                "row": row_index,
                "waypoint_index": lane_waypoint_index,
                "distance_m": round(next_lane_y - world_y, 4),
                "heading_start_deg": heading,
                "heading_end_deg": heading,
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((world_x, next_lane_y)),
            }
        )
        world_y = next_lane_y
        next_row_waypoint_index = len(waypoints)
        actions.append(
            {
                "kind": "turn_onto_next_row",
                "row": row_index,
                "waypoint_index": next_row_waypoint_index,
                "turn_deg": turn_degrees,
                "distance_m": 0.0,
                "heading_start_deg": heading,
                "heading_end_deg": heading + turn_degrees,
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((world_x, world_y)),
            }
        )
        heading += turn_degrees

    return {
        "robot": name,
        "partition": {
            "x_min": partition_x_min,
            "x_max": partition_x_min + PARTITION_WIDTH_METERS,
            "y_min": 0.0,
            "y_max": AREA_Y_METERS,
        },
        "start_center": round_point((x_min, lanes[0])),
        "start_heading_deg": 0.0,
        "lane_y_positions_m": [round(value, 4) for value in lanes],
        "lane_spacing_m": round(lanes[1] - lanes[0], 4),
        "waypoints": waypoints,
        "actions": actions,
    }


class MetadataRecorder:
    def __init__(self, path, plans):
        self.path = path
        self.lock = threading.Lock()
        self.current_actions = {
            name: {"step": None, "kind": "idle"}
            for name, _, _ in ROBOT_SPECS
        }
        self.completed = {name: False for name, _, _ in ROBOT_SPECS}
        self.primary_completed = {
            name: False for name, _, _ in ROBOT_SPECS
        }
        self.errors = {name: None for name, _, _ in ROBOT_SPECS}
        self.failed = {name: False for name, _, _ in ROBOT_SPECS}
        self.data = {
            "schema_version": 3,
            "recording_mode": "metadata_only",
            "run_started_utc": utc_timestamp(),
            "run_finished_utc": None,
            "configuration": {
                "robot_count": ROBOT_COUNT,
            "partition_width_m": round(PARTITION_WIDTH_METERS, 4),
            "robot_radius_m": ROBOT_RADIUS_METERS,
            "edge_margin_m": EDGE_MARGIN_METERS,
            "boundary_odometry_tolerance_m": (
                BOUNDARY_ODOMETRY_TOLERANCE_METERS
            ),
            "boundary_recovery_tolerance_m": (
                BOUNDARY_RECOVERY_TOLERANCE_METERS
            ),
            "partition_edge_margin_m": PARTITION_EDGE_MARGIN_METERS,
            "robot_clearance_m": ROBOT_CLEARANCE_METERS,
            "failed_robot_planning_clearance_m": (
                FAILED_ROBOT_PLANNING_CLEARANCE_METERS
            ),
            "active_robot_planning_clearance_m": (
                ACTIVE_ROBOT_PLANNING_CLEARANCE_METERS
            ),
            },
            "manual_failure": {
                "robot": FAILED_ROBOT_NAME,
                "trigger_key": "f",
                "behavior": "stationary_obstacle_then_reassign_remainder",
            },
            "failure_event": None,
            "takeover": {
                "assigned_robot": None,
                "assigned_utc": None,
                "source_waypoint_index": None,
                "skipped_blocked_waypoints": [],
                "completed": False,
            },
            "world_start": {
                name: {
                    "x": plan["start_center"]["x"],
                    "y": plan["start_center"]["y"],
                    "heading_deg": plan["start_heading_deg"],
                }
                for name, plan in plans.items()
            },
            "samples": [],
            "coverage": {"complete": False, "end_reason": None},
        }
        self.write()

    def _write_locked(self):
        temporary_path = self.path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2, sort_keys=True)
        temporary_path.replace(self.path)

    def write(self):
        with self.lock:
            self._write_locked()

    def update_robot(self, name, **values):
        with self.lock:
            if "completed" in values:
                self.completed[name] = bool(values["completed"])
            if "primary_completed" in values:
                self.primary_completed[name] = bool(
                    values["primary_completed"]
                )
            if "failed" in values:
                self.failed[name] = bool(values["failed"])
            if "error" in values:
                self.errors[name] = values["error"]

    def set_action(self, name, action):
        with self.lock:
            self.current_actions[name] = copy.deepcopy(action)
            self._write_locked()

    def record_failure(self, position, waypoint_index, elapsed_seconds):
        with self.lock:
            self.failed[FAILED_ROBOT_NAME] = True
            self.completed[FAILED_ROBOT_NAME] = True
            self.current_actions[FAILED_ROBOT_NAME] = {
                "step": waypoint_index,
                "kind": "failed_obstacle",
            }
            self.data["failure_event"] = {
                "robot": FAILED_ROBOT_NAME,
                "trigger": "terminal_key_f",
                "timestamp_utc": utc_timestamp(),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "position": round_point(position),
                "unfinished_waypoint_index": waypoint_index,
            }
            self._write_locked()

    def record_takeover_assignment(self, name, waypoint_index, route):
        with self.lock:
            self.data["takeover"].update(
                {
                    "assigned_robot": name,
                    "assigned_utc": utc_timestamp(),
                    "source_waypoint_index": waypoint_index,
                    "initial_transfer_route": [
                        round_point(point) for point in route
                    ],
                }
            )
            self._write_locked()

    def record_skipped_waypoint(self, waypoint_index, target):
        with self.lock:
            self.data["takeover"]["skipped_blocked_waypoints"].append(
                {
                    "source_waypoint_index": waypoint_index,
                    "target": round_point(target),
                    "reason": "inside_failed_robot_clearance",
                }
            )
            self._write_locked()

    def record_takeover_complete(self):
        with self.lock:
            self.data["takeover"]["completed"] = True
            self.data["takeover"]["completed_utc"] = utc_timestamp()
            self._write_locked()

    def record_positions(self, workers):
        sample = {"timestamp_utc": utc_timestamp(), "robots": {}}
        with self.lock:
            actions = copy.deepcopy(self.current_actions)
            failed = dict(self.failed)
        for worker in workers:
            position = worker.current_world_position()
            sample["robots"][worker.name] = {
                "position": {
                    "x": round(position[0], 4),
                    "y": round(position[1], 4),
                    "z": round(position[2], 4),
                    "heading_deg": round(worker.current_world_heading(), 3),
                },
                "action": actions[worker.name],
                "status": (
                    "failed_obstacle" if failed[worker.name] else "active"
                ),
            }
        with self.lock:
            self.data["samples"].append(sample)
            self._write_locked()

    def finish(self, complete, reason):
        with self.lock:
            self.data["coverage"].update(
                {"complete": complete, "end_reason": reason}
            )
            self.data["run_finished_utc"] = utc_timestamp()
            self._write_locked()


class FailureCoordinator:
    """Store failure state and assign takeover to the first finisher."""

    def __init__(self, failed_plan, recorder):
        self.failed_plan = failed_plan
        self.recorder = recorder
        self.lock = threading.Lock()
        self.failure_event = threading.Event()
        self.takeover_done = threading.Event()
        self.failure_position = None
        self.failure_waypoint_index = None
        self.finish_order = []
        self.assigned_robot = None

    def record_primary_completion(self, name):
        if name not in TAKEOVER_CANDIDATES:
            return
        with self.lock:
            if name not in self.finish_order:
                self.finish_order.append(name)

    def record_failure(self, position, waypoint_index, elapsed_seconds):
        with self.lock:
            if self.failure_event.is_set():
                return False
            self.failure_position = (position[0], position[1])
            self.failure_waypoint_index = waypoint_index
            self.recorder.record_failure(
                self.failure_position,
                waypoint_index,
                elapsed_seconds,
            )
            self.failure_event.set()
            return True

    def claim_takeover(self, name):
        with self.lock:
            if not self.failure_event.is_set() or self.assigned_robot:
                return False
            if not self.finish_order or self.finish_order[0] != name:
                return False
            self.assigned_robot = name
            return True

    def failed_obstacle(self):
        with self.lock:
            if self.failure_position is None:
                return []
            return [(self.failure_position, ROBOT_CLEARANCE_METERS)]


class MotionTelemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.position = (0.0, 0.0, 0.0)
        self.attitude = (0.0, 0.0, 0.0)
        self.position_ready = threading.Event()
        self.attitude_ready = threading.Event()

    def update_position(self, position_info):
        with self.lock:
            self.position = tuple(float(value) for value in position_info)
        self.position_ready.set()

    def update_attitude(self, attitude_info):
        with self.lock:
            self.attitude = tuple(float(value) for value in attitude_info)
        self.attitude_ready.set()

    def snapshot(self):
        with self.lock:
            return self.position, self.attitude

    def wait_until_ready(self, timeout):
        return (
            self.position_ready.wait(timeout)
            and self.attitude_ready.wait(timeout)
        )


class RobotWorker:
    def __init__(
        self,
        name,
        serial_number,
        plan,
        recorder,
        coordinator,
        args,
        stop_event,
        start_barrier,
    ):
        self.name = name
        self.serial_number = serial_number
        self.plan = plan
        self.recorder = recorder
        self.coordinator = coordinator
        self.args = args
        self.stop_event = stop_event
        self.start_barrier = start_barrier
        self.ep_robot = robot.Robot()
        self.chassis = None
        self.initialized = False
        self.position_subscribed = False
        self.attitude_subscribed = False
        self.telemetry = MotionTelemetry()
        self.initial_yaw = None
        self.initial_position = None
        self.world_yaw = None
        self.current_waypoint_index = 0
        self.failed_event = threading.Event()
        self.motion_command_lock = threading.Lock()
        self.peers = []

    def set_peers(self, workers):
        self.peers = [worker for worker in workers if worker is not self]

    def connect(self):
        initialized = bool(
            self.ep_robot.initialize(conn_type="sta", sn=self.serial_number)
        )
        if not initialized:
            raise RuntimeError(
                "Could not connect to {0} ({1})".format(
                    self.name,
                    self.serial_number,
                )
            )
        self.initialized = True
        self.chassis = self.ep_robot.chassis
        self.position_subscribed = bool(
            self.chassis.sub_position(
                cs=0,
                freq=20,
                callback=self.telemetry.update_position,
            )
        )
        self.attitude_subscribed = bool(
            self.chassis.sub_attitude(
                freq=20,
                callback=self.telemetry.update_attitude,
            )
        )
        if not self.position_subscribed or not self.attitude_subscribed:
            raise RuntimeError(
                "Could not subscribe to telemetry for {0}".format(self.name)
            )
        if not self.telemetry.wait_until_ready(timeout=5.0):
            raise RuntimeError(
                "Timed out waiting for telemetry for {0}".format(self.name)
            )
        position, attitude = self.telemetry.snapshot()
        self.initial_position = position
        self.initial_yaw = attitude[0]
        print(
            "{0}: startup yaw is {1:.2f} deg.".format(
                self.name,
                self.initial_yaw,
            )
        )

    def set_world_yaw(self, world_yaw):
        self.world_yaw = normalize_angle_degrees(world_yaw)
        print(
            "{0}: arena +x yaw target is {1:.2f} deg.".format(
                self.name,
                self.world_yaw,
            )
        )

    def drive_speed(self, x, y, z, timeout=None):
        with self.motion_command_lock:
            if self.failed_event.is_set():
                x = y = z = 0.0
            kwargs = {"x": x, "y": y, "z": z}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self.chassis.drive_speed(**kwargs)

    def command_stop(self):
        self.drive_speed(
            x=0.0,
            y=0.0,
            z=0.0,
            timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
        )

    def raise_if_failed(self):
        if self.failed_event.is_set():
            raise ManualRobotFailure()

    def fail_as_obstacle(self, elapsed_seconds):
        if (
            self.name != FAILED_ROBOT_NAME
            or self.failed_event.is_set()
            or self.recorder.primary_completed[self.name]
        ):
            return False
        self.failed_event.set()
        self.command_stop()
        position = self.current_world_position()
        recorded = self.coordinator.record_failure(
            position,
            self.current_waypoint_index,
            elapsed_seconds,
        )
        if recorded:
            print(
                "\nrobot_2 failed at ({0:.3f}, {1:.3f}) m. It is now a "
                "stationary obstacle; takeover begins after robot_1 or "
                "robot_3 completes its own partition.".format(
                    position[0],
                    position[1],
                )
            )
        return recorded

    def motion_timeout(self, distance=0.0, turn_degrees=0.0):
        expected_seconds = (
            distance / self.args.linear_speed
            + abs(turn_degrees) / self.args.turn_speed
        )
        return (
            expected_seconds * MOTION_TIMEOUT_MULTIPLIER
            + MOTION_TIMEOUT_MARGIN_SECONDS
        )

    def current_world_position(self):
        position, _ = self.telemetry.snapshot()
        if self.initial_position is None or self.world_yaw is None:
            return (0.0, 0.0, 0.0)
        start = self.plan["start_center"]
        delta_x = position[0] - self.initial_position[0]
        delta_y = position[1] - self.initial_position[1]
        yaw_radians = math.radians(self.world_yaw)
        world_delta_x = (
            delta_x * math.cos(yaw_radians)
            + delta_y * math.sin(yaw_radians)
        )
        world_delta_y = (
            -delta_x * math.sin(yaw_radians)
            + delta_y * math.cos(yaw_radians)
        )
        return (
            start["x"] + world_delta_x,
            start["y"] + world_delta_y,
            position[2] - self.initial_position[2],
        )

    def current_world_heading(self):
        _, attitude = self.telemetry.snapshot()
        if self.world_yaw is None:
            return 0.0
        return normalize_angle_degrees(attitude[0] - self.world_yaw)

    def planned_yaw(self, world_heading):
        return normalize_angle_degrees(self.world_yaw + world_heading)

    def rotate_to_yaw(self, target_yaw):
        self.raise_if_failed()
        _, attitude = self.telemetry.snapshot()
        target_yaw = normalize_angle_degrees(target_yaw)
        deadline = time.monotonic() + self.motion_timeout(
            turn_degrees=normalize_angle_degrees(target_yaw - attitude[0])
        )
        while not self.stop_event.is_set():
            self.raise_if_failed()
            _, attitude = self.telemetry.snapshot()
            error = normalize_angle_degrees(target_yaw - attitude[0])
            if abs(error) <= TURN_TOLERANCE_DEGREES:
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("rotation timeout for {0}".format(self.name))
            turn_speed = min(
                self.args.turn_speed,
                max(MIN_TURN_SPEED_DPS, abs(error) * 0.8),
            )
            self.drive_speed(
                x=0.0,
                y=0.0,
                z=math.copysign(turn_speed, error),
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)
        raise RuntimeError("rotation stopped before completion")

    def nearest_peer_distance(self, world_position):
        distances = []
        for peer in self.peers:
            peer_position = peer.current_world_position()
            distances.append(point_distance(world_position, peer_position))
        return min(distances) if distances else float("inf")

    def planning_obstacles(self):
        """Return live peer positions with execution-error planning buffers."""
        obstacles = []
        for peer in self.peers:
            clearance = (
                FAILED_ROBOT_PLANNING_CLEARANCE_METERS
                if (
                    peer.name == FAILED_ROBOT_NAME
                    and peer.failed_event.is_set()
                )
                else ACTIVE_ROBOT_PLANNING_CLEARANCE_METERS
            )
            obstacles.append(
                (
                    peer.current_world_position()[:2],
                    clearance,
                )
            )
        return obstacles

    def current_failed_robot_position(self):
        """Use live failed-robot telemetry instead of its stale F-key pose."""
        for peer in self.peers:
            if (
                peer.name == FAILED_ROBOT_NAME
                and peer.failed_event.is_set()
            ):
                return peer.current_world_position()[:2]
        return self.coordinator.failure_position

    def drive_forward(self, distance, target_yaw):
        self.raise_if_failed()
        position, _ = self.telemetry.snapshot()
        start_x, start_y, _ = position
        target_yaw = normalize_angle_degrees(target_yaw)
        heading_radians = math.radians(target_yaw)
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        right_x = -forward_y
        right_y = forward_x
        last_correction_travelled = 0.0
        deadline = time.monotonic() + self.motion_timeout(distance=distance)
        peer_wait_started = None

        while not self.stop_event.is_set():
            self.raise_if_failed()
            position, attitude = self.telemetry.snapshot()
            delta_x = position[0] - start_x
            delta_y = position[1] - start_y
            travelled = delta_x * forward_x + delta_y * forward_y
            remaining = distance - travelled
            if remaining <= DISTANCE_TOLERANCE_METERS:
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                return {
                    "requested_distance_m": round(distance, 4),
                    "travelled_distance_m": round(travelled, 4),
                    "cross_track_error_m": round(
                        delta_x * right_x + delta_y * right_y,
                        4,
                    ),
                }
            if time.monotonic() >= deadline:
                raise RuntimeError("forward timeout for {0}".format(self.name))

            peer_distance = self.nearest_peer_distance(
                self.current_world_position()
            )
            if peer_distance < ROBOT_CLEARANCE_METERS:
                self.command_stop()
                if peer_wait_started is None:
                    peer_wait_started = time.monotonic()
                if (
                    time.monotonic() - peer_wait_started
                    > PEER_WAIT_TIMEOUT_SECONDS
                ):
                    raise RuntimeError(
                        "{0} blocked by another robot at {1:.3f} m".format(
                            self.name,
                            peer_distance,
                        )
                    )
                time.sleep(CONTROL_PERIOD_SECONDS)
                continue
            peer_wait_started = None

            yaw_error = normalize_angle_degrees(target_yaw - attitude[0])
            if (
                abs(yaw_error) > HEADING_DRIFT_TOLERANCE_DEGREES
                and travelled - last_correction_travelled
                >= MIN_DISTANCE_BETWEEN_HEADING_CORRECTIONS_METERS
            ):
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                self.rotate_to_yaw(target_yaw)
                last_correction_travelled = travelled
                continue
            forward_speed = min(
                self.args.linear_speed,
                max(MIN_LINEAR_SPEED_MPS, remaining * 0.8),
            )
            self.drive_speed(
                x=forward_speed,
                y=0.0,
                z=0.0,
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)
        raise RuntimeError("forward motion stopped before completion")

    def move_to_point(self, target, action):
        current = self.current_world_position()
        distance = point_distance(current, target)
        if distance <= DISTANCE_TOLERANCE_METERS:
            return
        heading = math.degrees(
            math.atan2(target[1] - current[1], target[0] - current[0])
        )
        action_record = copy.deepcopy(action)
        action_record.update(
            {
                "world_start": round_point(current),
                "world_end": round_point(target),
                "heading_deg": round(heading, 3),
                "distance_m": round(distance, 4),
                "started_utc": utc_timestamp(),
            }
        )
        self.recorder.set_action(self.name, action_record)
        print(
            "{0}: {1} to ({2:.3f}, {3:.3f}), distance={4:.3f} m.".format(
                self.name,
                action["kind"],
                target[0],
                target[1],
                distance,
            )
        )
        self.command_stop()
        time.sleep(MOTION_SETTLE_SECONDS)
        target_yaw = self.planned_yaw(heading)
        self.rotate_to_yaw(target_yaw)
        self.drive_forward(distance, target_yaw)

    def execute_primary_action(self, action_index, planned_action):
        """Execute normal coverage exactly as centralized.py does."""
        self.current_waypoint_index = planned_action["waypoint_index"]
        action_record = copy.deepcopy(planned_action)
        action_record.update(
            {
                "step": action_index,
                "mode": "primary",
                "started_utc": utc_timestamp(),
            }
        )
        self.recorder.set_action(self.name, action_record)

        self.command_stop()
        time.sleep(MOTION_SETTLE_SECONDS)
        if planned_action.get("turn_deg", 0.0):
            print(
                "{0}: {1}, heading {2:.1f}->{3:.1f} deg.".format(
                    self.name,
                    planned_action["kind"],
                    planned_action["heading_start_deg"],
                    planned_action["heading_end_deg"],
                )
            )
            self.rotate_to_yaw(
                self.planned_yaw(planned_action["heading_end_deg"])
            )
            return

        print(
            "{0}: {1}, distance={2:.3f} m, heading={3:.1f} deg.".format(
                self.name,
                planned_action["kind"],
                planned_action["distance_m"],
                planned_action["heading_start_deg"],
            )
        )
        target_yaw = self.planned_yaw(
            planned_action["heading_start_deg"]
        )
        _, attitude = self.telemetry.snapshot()
        if (
            abs(normalize_angle_degrees(target_yaw - attitude[0]))
            > TURN_TOLERANCE_DEGREES
        ):
            self.rotate_to_yaw(target_yaw)
        self.drive_forward(planned_action["distance_m"], target_yaw)

    def execute_takeover_waypoint(self, waypoint_index, waypoint):
        """Route one remaining robot-2 waypoint around live obstacles."""
        target = (
            waypoint["world_end"]["x"],
            waypoint["world_end"]["y"],
        )
        route = plan_clear_path(
            self.current_world_position(),
            target,
            self.planning_obstacles(),
        )
        for segment_index, route_point in enumerate(route):
            self.move_to_point(
                route_point,
                {
                    "step": waypoint_index,
                    "kind": (
                        waypoint["kind"]
                        if len(route) == 1
                        else "takeover_obstacle_detour"
                    ),
                    "mode": "takeover",
                    "route_segment": segment_index,
                    "route_segment_count": len(route),
                    "source_robot": FAILED_ROBOT_NAME,
                },
            )

    def run_primary_plan(self):
        for action_index, planned_action in enumerate(self.plan["actions"]):
            if self.stop_event.is_set():
                raise RuntimeError("Stopped because another worker failed")
            self.execute_primary_action(action_index, planned_action)
        self.current_waypoint_index = len(self.plan["waypoints"])
        self.recorder.update_robot(
            self.name,
            primary_completed=True,
            completed=True,
        )
        self.coordinator.record_primary_completion(self.name)

    def run_takeover(self):
        start_index = self.coordinator.failure_waypoint_index
        waypoints = self.coordinator.failed_plan["waypoints"]

        while start_index < len(waypoints):
            failed_position = self.current_failed_robot_position()
            target_record = waypoints[start_index]["world_end"]
            target = (target_record["x"], target_record["y"])
            if (
                point_distance(target, failed_position)
                >= FAILED_ROBOT_PLANNING_CLEARANCE_METERS
            ):
                break
            self.recorder.record_skipped_waypoint(start_index, target)
            print(
                "{0}: skipping robot_2 waypoint {1}; its target is inside "
                "the failed-robot clearance circle.".format(
                    self.name,
                    start_index,
                )
            )
            start_index += 1

        if start_index >= len(waypoints):
            self.recorder.record_takeover_assignment(
                self.name,
                start_index,
                [],
            )
            self.recorder.record_takeover_complete()
            self.coordinator.takeover_done.set()
            return

        first_target_record = waypoints[start_index]["world_end"]
        first_target = (
            first_target_record["x"],
            first_target_record["y"],
        )
        initial_route = plan_clear_path(
            self.current_world_position(),
            first_target,
            self.planning_obstacles(),
        )
        self.recorder.record_takeover_assignment(
            self.name,
            start_index,
            initial_route,
        )
        print(
            "{0}: taking over robot_2 at waypoint {1}; planned clearance "
            "is {2:.3f} m center-to-center.".format(
                self.name,
                start_index,
                FAILED_ROBOT_PLANNING_CLEARANCE_METERS,
            )
        )
        for source_index in range(start_index, len(waypoints)):
            waypoint = waypoints[source_index]
            target_record = waypoint["world_end"]
            target = (target_record["x"], target_record["y"])
            failed_position = self.current_failed_robot_position()
            if (
                point_distance(target, failed_position)
                < FAILED_ROBOT_PLANNING_CLEARANCE_METERS
            ):
                self.recorder.record_skipped_waypoint(source_index, target)
                continue
            self.execute_takeover_waypoint(source_index, waypoint)
        self.recorder.record_takeover_complete()
        self.coordinator.takeover_done.set()

    def run(self):
        try:
            self.start_barrier.wait()
            self.run_primary_plan()
            if self.name not in TAKEOVER_CANDIDATES:
                return
            while not self.stop_event.is_set():
                if self.coordinator.claim_takeover(self.name):
                    self.run_takeover()
                    return
                if self.coordinator.takeover_done.is_set():
                    return
                if (
                    not self.coordinator.failure_event.is_set()
                    and all(self.recorder.primary_completed.values())
                ):
                    return
                time.sleep(CONTROL_PERIOD_SECONDS)
        except ManualRobotFailure:
            self.recorder.update_robot(
                self.name,
                failed=True,
                completed=True,
            )
        except Exception as error:
            print("{0} failed: {1}".format(self.name, error))
            self.recorder.update_robot(self.name, error=str(error))
            self.stop_event.set()
        finally:
            self.stop_motion()

    def stop_motion(self):
        if self.chassis is not None:
            try:
                self.drive_speed(x=0.0, y=0.0, z=0.0)
            except Exception:
                pass

    def close(self):
        self.stop_motion()
        if self.attitude_subscribed and self.chassis is not None:
            try:
                self.chassis.unsub_attitude()
            except Exception:
                pass
        if self.position_subscribed and self.chassis is not None:
            try:
                self.chassis.unsub_position()
            except Exception:
                pass
        if self.initialized:
            try:
                self.ep_robot.close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run three equal lawnmower partitions. Press F to fail robot_2; "
            "the first of robot_1 or robot_3 to finish takes over."
        )
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=DEFAULT_LINEAR_SPEED_MPS,
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=DEFAULT_TURN_SPEED_DPS,
    )
    parser.add_argument(
        "--world-yaw",
        type=float,
        default=None,
        help=(
            "Optional shared SDK yaw for arena +x. By default, each robot "
            "uses its own startup yaw to avoid cross-robot yaw offsets."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "centralized_failure_runs",
    )
    args = parser.parse_args()
    if not 0.08 <= args.linear_speed <= 1.0:
        raise SystemExit("--linear-speed must be between 0.08 and 1.0 m/s")
    if not 10.0 <= args.turn_speed <= 540.0:
        raise SystemExit("--turn-speed must be between 10 and 540 degrees/s")
    return args


def main():
    media.LiveView = NoMediaLiveView
    args = parse_args()
    run_directory = (
        args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    plans = {
        name: build_robot_plan(name, partition_x_min)
        for name, _, partition_x_min in ROBOT_SPECS
    }
    recorder = MetadataRecorder(run_directory / "metadata.json", plans)
    coordinator = FailureCoordinator(plans[FAILED_ROBOT_NAME], recorder)
    stop_event = threading.Event()
    start_barrier = threading.Barrier(len(ROBOT_SPECS) + 1)
    workers = []
    motion_threads = []

    print("Output: {0}".format(run_directory))
    print("Metadata-only mode: cameras and image saving are disabled.")
    print(
        "Partitions are {0:.3f} m wide; live robot clearance is {1:.3f} m. "
        "Takeover routes reserve {2:.3f} m around failed robot_2.".format(
            PARTITION_WIDTH_METERS,
            ROBOT_CLEARANCE_METERS,
            FAILED_ROBOT_PLANNING_CLEARANCE_METERS,
        )
    )
    for name, _, _ in ROBOT_SPECS:
        start = plans[name]["start_center"]
        print(
            "Place {0} center at ({1:.3f}, {2:.3f}) m.".format(
                name,
                start["x"],
                start["y"],
            )
        )
    print("Point all robots along arena +x. Press F to fail robot_2.")

    try:
        for name, serial_number, _ in ROBOT_SPECS:
            worker = RobotWorker(
                name=name,
                serial_number=serial_number,
                plan=plans[name],
                recorder=recorder,
                coordinator=coordinator,
                args=args,
                stop_event=stop_event,
                start_barrier=start_barrier,
            )
            worker.connect()
            workers.append(worker)

        for worker in workers:
            world_yaw = (
                args.world_yaw
                if args.world_yaw is not None
                else worker.initial_yaw
            )
            worker.set_world_yaw(world_yaw)
            worker.set_peers(workers)
        recorder.record_positions(workers)

        for worker in workers:
            thread = threading.Thread(
                target=worker.run,
                name="{0}-motion".format(worker.name),
            )
            motion_threads.append(thread)
            thread.start()

        start_barrier.wait()
        motion_started = time.monotonic()
        failed_worker = next(
            worker
            for worker in workers
            if worker.name == FAILED_ROBOT_NAME
        )
        next_sample = time.monotonic() + METADATA_SAMPLE_PERIOD_SECONDS
        while any(thread.is_alive() for thread in motion_threads):
            for thread in motion_threads:
                thread.join(timeout=0.02)
            now = time.monotonic()
            while msvcrt.kbhit():
                if msvcrt.getwch().lower() == "f":
                    if not failed_worker.fail_as_obstacle(
                        now - motion_started
                    ):
                        print("robot_2 is already failed or has finished.")
            if now >= next_sample:
                recorder.record_positions(workers)
                next_sample = now + METADATA_SAMPLE_PERIOD_SECONDS
            if stop_event.is_set():
                for worker in workers:
                    worker.stop_motion()

        recorder.record_positions(workers)
        failure_occurred = coordinator.failure_event.is_set()
        complete = (
            not stop_event.is_set()
            and recorder.primary_completed["robot_1"]
            and recorder.primary_completed["robot_3"]
            and (
                coordinator.takeover_done.is_set()
                if failure_occurred
                else recorder.primary_completed["robot_2"]
            )
        )
        if complete and failure_occurred:
            reason = (
                "{0} completed robot_2's reachable remaining waypoints.".format(
                    coordinator.assigned_robot
                )
            )
        elif complete:
            reason = "All three partition plans completed without failure."
        else:
            errors = [
                "{0}: {1}".format(name, recorder.errors[name])
                for name, _, _ in ROBOT_SPECS
                if recorder.errors[name]
            ]
            reason = "Coverage stopped before completion."
            if errors:
                reason += " " + " | ".join(errors)
        recorder.finish(complete, reason)
        print(reason)

    except KeyboardInterrupt:
        stop_event.set()
        for worker in workers:
            worker.stop_motion()
        try:
            start_barrier.abort()
        except threading.BrokenBarrierError:
            pass
        for thread in motion_threads:
            thread.join(timeout=3.0)
        recorder.finish(False, "Interrupted by user.")
        print("Interrupted. All robots are stopping.")
    except Exception as error:
        stop_event.set()
        for worker in workers:
            worker.stop_motion()
        try:
            start_barrier.abort()
        except threading.BrokenBarrierError:
            pass
        recorder.finish(False, "Run failed: {0}".format(error))
        print("Run failed: {0}".format(error))
    finally:
        stop_event.set()
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
