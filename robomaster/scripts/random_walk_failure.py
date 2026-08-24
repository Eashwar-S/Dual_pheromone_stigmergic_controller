# -*-coding:utf-8-*-
"""Memoryless three-RoboMaster random-walk coverage with a failed obstacle.

All robots share the entire arena. At each normal decision, each robot
randomly chooses one direction: left, straight, or right. Choices that
would leave the arena are filtered out. Robot collisions use a distinct
90-degree turn-and-forward escape maneuver; no pheromone logic is used.
Pressing F in the terminal abruptly stops robot 2, which then remains a
stationary obstacle for the other robots.
"""

import argparse
from datetime import datetime
import json
import math
import msvcrt
from pathlib import Path
import random
import sys
import threading
import time
import types


# Prefer this SDK checkout over an older robomaster package in site-packages.
SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

# The SDK imports libmedia_codec while loading its camera module, even when no
# camera stream is requested. This metadata-only experiment replaces LiveView
# before any Robot is constructed, so a placeholder prevents the unused native
# H.264/Opus extension from loading into the motion-control process.
sys.modules["libmedia_codec"] = types.ModuleType("libmedia_codec")
from robomaster import media, robot


class NoMediaLiveView:
    """Satisfy the SDK camera module without creating media decoders."""

    def __init__(self, ep_robot):
        self._robot = ep_robot

    def start_video_stream(self, display=False, addr=None, ip_proto="tcp"):
        return True

    def stop_video_stream(self):
        return True

    def stop(self):
        pass


AREA_X_METERS = 2.0
AREA_Y_METERS = 3.8
ROBOT_COUNT = 3
INITIAL_ROBOT_SPACING_METERS = AREA_Y_METERS / ROBOT_COUNT
# Retained for older scripts that import this constant; random_walk.py no
# longer uses it as a movement partition.
PARTITION_WIDTH_METERS = INITIAL_ROBOT_SPACING_METERS
ROBOT_SPECS = (
    ("robot_1", "3JKCH8800100VW", 0.0),
    ("robot_2", "3JKCH8800100RC", INITIAL_ROBOT_SPACING_METERS),
    ("robot_3", "3JKCH8800100VZ", 2 * INITIAL_ROBOT_SPACING_METERS),
)
FAILED_ROBOT_NAME = "robot_2"
ROBOT_RADIUS_METERS = 8.0 * 0.0254
EDGE_MARGIN_METERS = 0.05
BOUNDARY_ODOMETRY_TOLERANCE_METERS = 0.02
ROBOT_SAFETY_DISTANCE_METERS = 2.0 * ROBOT_RADIUS_METERS
COLLISION_AVOIDANCE_TURN_DEGREES = 90.0
BOUNDARY_LOOKAHEAD_METERS = 0.03

DEFAULT_LINEAR_SPEED_MPS = 0.2
DEFAULT_TURN_SPEED_DPS = 30.0
DEFAULT_DURATION_SECONDS = 300.0
DEFAULT_STEP_DISTANCE_METERS = 0.25
ACTION_TURN_DEGREES = 30.0
METADATA_SAMPLE_PERIOD_SECONDS = 0.25
CONTROL_PERIOD_SECONDS = 0.05
DRIVE_COMMAND_TIMEOUT_SECONDS = 0.3
MOTION_SETTLE_SECONDS = 0.4
TURN_TOLERANCE_DEGREES = 2.0
HEADING_DRIFT_TOLERANCE_DEGREES = 4.0
DISTANCE_TOLERANCE_METERS = 0.025
MIN_TURN_SPEED_DPS = 5.0
MIN_LINEAR_SPEED_MPS = 0.08
MOTION_TIMEOUT_MULTIPLIER = 4.0
MOTION_TIMEOUT_MARGIN_SECONDS = 10.0
COVERAGE_CELL_SIZE_METERS = 0.05

ACTION_TURNS = {
    "left": ACTION_TURN_DEGREES,
    "straight": 0.05,
    "right": -ACTION_TURN_DEGREES,
}


class ManualRobotFailure(Exception):
    """End one robot's motion without stopping the experiment."""


def utc_timestamp():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def round_point(point):
    return {"x": round(point[0], 4), "y": round(point[1], 4)}


def normalize_angle_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def build_robot_start(name, start_y):
    x_min = EDGE_MARGIN_METERS
    y_min = start_y + EDGE_MARGIN_METERS
    return {
        "robot": name,
        "partition": {
            "x_min": 0.0,
            "x_max": AREA_X_METERS,
            "y_min": 0.0,
            "y_max": AREA_Y_METERS,
        },
        "start_center": round_point((x_min, y_min)),
        "start_heading_deg": 0.0,
    }


def boundary_violation(point, partition):
    """Distance outside the permitted center-position rectangle."""
    x_min = partition["x_min"] + EDGE_MARGIN_METERS
    x_max = partition["x_max"] - EDGE_MARGIN_METERS
    y_min = partition["y_min"] + EDGE_MARGIN_METERS
    y_max = partition["y_max"] - EDGE_MARGIN_METERS
    outside_x = max(x_min - point[0], 0.0, point[0] - x_max)
    outside_y = max(y_min - point[1], 0.0, point[1] - y_max)
    return math.hypot(outside_x, outside_y)


def inside_bounds(point, partition, tolerance=0.0):
    return (
        boundary_violation(point, partition) <= tolerance
    )


def project_point(point, heading_deg, distance):
    radians = math.radians(heading_deg)
    return (
        point[0] + distance * math.cos(radians),
        point[1] + distance * math.sin(radians),
    )


class CoverageTracker:
    def __init__(self, cell_size, robot_radius):
        self.cell_size = cell_size
        self.robot_radius = robot_radius
        self.covered_cells = set()
        self.total_area = AREA_X_METERS * AREA_Y_METERS

    def add_robot_position(self, position):
        x, y = position[0], position[1]
        radius_cells = int(math.ceil(self.robot_radius / self.cell_size))
        center_col = int(math.floor(x / self.cell_size))
        center_row = int(math.floor(y / self.cell_size))

        for col in range(center_col - radius_cells, center_col + radius_cells + 1):
            for row in range(center_row - radius_cells, center_row + radius_cells + 1):
                cell_x = (col + 0.5) * self.cell_size
                cell_y = (row + 0.5) * self.cell_size
                if not (0.0 <= cell_x <= AREA_X_METERS and 0.0 <= cell_y <= AREA_Y_METERS):
                    continue
                if math.hypot(cell_x - x, cell_y - y) <= self.robot_radius:
                    self.covered_cells.add((col, row))

    def summary(self):
        covered_area = min(
            len(self.covered_cells) * self.cell_size * self.cell_size,
            self.total_area,
        )
        return {
            "covered_area_m2": round(covered_area, 4),
            "total_area_m2": round(self.total_area, 4),
            "percent": round(100.0 * covered_area / self.total_area, 2),
        }


class MetadataRecorder:
    def __init__(self, path, starts, args):
        self.path = path
        self.lock = threading.Lock()
        self.current_actions = {
            name: {"name": "idle", "turn_deg": 0.0}
            for name, _, _ in ROBOT_SPECS
        }
        self.completed = {name: False for name, _, _ in ROBOT_SPECS}
        self.errors = {name: None for name, _, _ in ROBOT_SPECS}
        self.failed = {name: False for name, _, _ in ROBOT_SPECS}
        self.coverage = CoverageTracker(COVERAGE_CELL_SIZE_METERS, ROBOT_RADIUS_METERS)
        self.data = {
            "schema_version": 1,
            "run_started_utc": utc_timestamp(),
            "run_finished_utc": None,
            "recording_mode": "metadata_only",
            "manual_failure": {
                "robot": FAILED_ROBOT_NAME,
                "trigger_key": "f",
                "behavior": "abrupt_stop_stationary_obstacle",
            },
            "failure_event": None,
            "world_start": {
                name: {
                    "x": start["start_center"]["x"],
                    "y": start["start_center"]["y"],
                    "heading_deg": start["start_heading_deg"],
                }
                for name, start in starts.items()
            },
            "samples": [],
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

    def set_action(self, name, action):
        with self.lock:
            self.current_actions[name] = action

    def record_manual_failure(self, name, action, elapsed_seconds):
        with self.lock:
            self.current_actions[name] = action
            self.completed[name] = True
            self.failed[name] = True
            self.data["failure_event"] = {
                "robot": name,
                "trigger": "terminal_key_f",
                "elapsed_seconds": round(elapsed_seconds, 3),
                "timestamp_utc": action["failed_utc"],
            }
            self._write_locked()

    def update_robot(self, name, completed=None, error=None, failed=None):
        with self.lock:
            if completed is not None:
                self.completed[name] = bool(completed)
            if error is not None:
                self.errors[name] = error
            if failed is not None:
                self.failed[name] = bool(failed)

    def record_positions(self, workers):
        timestamp = utc_timestamp()
        sample = {"timestamp_utc": timestamp, "robots": {}}

        with self.lock:
            actions = dict(self.current_actions)
            failed = dict(self.failed)

        for worker in workers:
            position = worker.current_world_position()
            self.coverage.add_robot_position(position)
            sample["robots"][worker.name] = {
                "position": {
                    "x": round(position[0], 4),
                    "y": round(position[1], 4),
                    "z": round(position[2], 4),
                    "heading_deg": round(worker.current_world_heading(), 3),
                },
                "action": actions[worker.name],
                "status": (
                    "failed_obstacle"
                    if failed[worker.name]
                    else "active"
                ),
            }

        sample["coverage"] = self.coverage.summary()
        with self.lock:
            self.data["samples"].append(sample)
            self._write_locked()

    def finish(self, reason):
        with self.lock:
            self.data["run_finished_utc"] = utc_timestamp()
            self.data["end_reason"] = reason
            self._write_locked()


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
        position_ok = self.position_ready.wait(timeout)
        attitude_ok = self.attitude_ready.wait(timeout)
        return position_ok and attitude_ok


class RobotWorker:
    def __init__(
        self,
        name,
        serial_number,
        start,
        recorder,
        args,
        stop_event,
        start_barrier,
    ):
        self.name = name
        self.serial_number = serial_number
        self.start = start
        self.recorder = recorder
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
        self.relative_heading = 0.0
        self.random = random.Random(args.seed + sum(ord(char) for char in name))
        self.peers = []
        self.failed_event = threading.Event()
        self.motion_command_lock = threading.Lock()

    def set_peers(self, workers):
        self.peers = [worker for worker in workers if worker is not self]

    def connect(self):
        initialized = bool(
            self.ep_robot.initialize(conn_type="sta", sn=self.serial_number)
        )
        if not initialized:
            raise RuntimeError(
                "Could not connect to {0} ({1})".format(self.name, self.serial_number)
            )

        self.initialized = True
        self.chassis = self.ep_robot.chassis
        self.position_subscribed = bool(
            self.chassis.sub_position(cs=0, freq=20, callback=self.telemetry.update_position)
        )
        self.attitude_subscribed = bool(
            self.chassis.sub_attitude(freq=20, callback=self.telemetry.update_attitude)
        )
        if not self.position_subscribed or not self.attitude_subscribed:
            raise RuntimeError("Could not subscribe to motion telemetry for {0}".format(self.name))
        if not self.telemetry.wait_until_ready(timeout=5.0):
            raise RuntimeError("Timed out waiting for motion telemetry for {0}".format(self.name))

        position, attitude = self.telemetry.snapshot()
        self.initial_position = position
        self.initial_yaw = attitude[0]
        print("{0}: startup yaw is {1:.2f} deg.".format(self.name, self.initial_yaw))

    def set_world_yaw(self, world_yaw):
        self.world_yaw = normalize_angle_degrees(world_yaw)
        print("{0}: arena +x yaw target is {1:.2f} deg.".format(self.name, self.world_yaw))

    def motion_timeout(self, distance=0.0, turn_degrees=0.0):
        expected_seconds = distance / self.args.linear_speed + abs(turn_degrees) / self.args.turn_speed
        return expected_seconds * MOTION_TIMEOUT_MULTIPLIER + MOTION_TIMEOUT_MARGIN_SECONDS

    def drive_speed(self, x, y, z, timeout=None):
        """Serialize commands and never permit motion after manual failure."""
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
        """Abruptly stop this robot and permanently reject later motion."""
        if self.failed_event.is_set():
            return False
        self.failed_event.set()
        self.command_stop()
        action = {
            "name": "failed_obstacle",
            "turn_deg": 0.0,
            "distance_m": 0.0,
            "trigger": "terminal_key_f",
            "elapsed_seconds": round(elapsed_seconds, 3),
            "failed_utc": utc_timestamp(),
        }
        self.recorder.record_manual_failure(
            self.name,
            action,
            elapsed_seconds,
        )
        local_time = datetime.now().isoformat(timespec="milliseconds")
        print(
            "\n{0}: F pressed at {1} local time ({2:.3f} s after motion "
            "started). Motion stopped abruptly; robot remains a stationary "
            "obstacle.".format(
                self.name,
                local_time,
                elapsed_seconds,
            )
        )
        return True

    def current_world_position(self):
        position, _ = self.telemetry.snapshot()
        if self.initial_position is None or self.world_yaw is None:
            return (0.0, 0.0, 0.0)

        start = self.start["start_center"]
        delta_x = position[0] - self.initial_position[0]
        delta_y = position[1] - self.initial_position[1]
        yaw_radians = math.radians(self.world_yaw)
        world_delta_x = delta_x * math.cos(yaw_radians) + delta_y * math.sin(yaw_radians)
        world_delta_y = -delta_x * math.sin(yaw_radians) + delta_y * math.cos(yaw_radians)
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

    def target_sdk_yaw(self, relative_heading):
        return normalize_angle_degrees(self.world_yaw + relative_heading)

    def peer_distance(self, point):
        distances = []
        for peer in self.peers:
            peer_point = peer.current_world_position()
            distances.append(
                math.hypot(
                    point[0] - peer_point[0],
                    point[1] - peer_point[1],
                )
            )
        return min(distances) if distances else float("inf")

    def action_endpoint(self, action):
        return project_point(
            self.current_world_position(),
            action[2],
            action[3],
        )

    def action_is_peer_safe(self, action):
        current = self.current_world_position()
        endpoint = self.action_endpoint(action)
        segment_x = endpoint[0] - current[0]
        segment_y = endpoint[1] - current[1]
        segment_length_squared = (
            segment_x * segment_x + segment_y * segment_y
        )

        for peer in self.peers:
            peer_point = peer.current_world_position()
            if segment_length_squared <= 0.0:
                closest_x, closest_y = current[0], current[1]
            else:
                projection = (
                    (peer_point[0] - current[0]) * segment_x
                    + (peer_point[1] - current[1]) * segment_y
                ) / segment_length_squared
                projection = max(0.0, min(1.0, projection))
                closest_x = current[0] + projection * segment_x
                closest_y = current[1] + projection * segment_y
            if (
                math.hypot(
                    closest_x - peer_point[0],
                    closest_y - peer_point[1],
                )
                < ROBOT_SAFETY_DISTANCE_METERS
            ):
                return False
        return True

    def choose_collision_avoidance_action(self):
        """Choose a boundary-safe 90-degree collision escape."""
        current = self.current_world_position()
        boundary = self.start["partition"]
        current_heading = self.current_world_heading()
        current_peer_distance = self.peer_distance(current)
        moving_candidates = []
        rotation_candidates = []

        for action_name, turn_degrees in (
            ("left", COLLISION_AVOIDANCE_TURN_DEGREES),
            ("right", -COLLISION_AVOIDANCE_TURN_DEGREES),
        ):
            next_heading = normalize_angle_degrees(
                current_heading + turn_degrees
            )
            endpoint = project_point(
                current,
                next_heading,
                self.args.step_distance,
            )
            action = (
                action_name,
                turn_degrees,
                next_heading,
                self.args.step_distance,
            )
            endpoint_peer_distance = self.peer_distance(endpoint)
            rotation_candidates.append(
                (
                    -endpoint_peer_distance,
                    action_name,
                    turn_degrees,
                    next_heading,
                )
            )

            if not inside_bounds(
                endpoint,
                boundary,
                tolerance=BOUNDARY_ODOMETRY_TOLERANCE_METERS,
            ):
                continue
            if current_peer_distance < ROBOT_SAFETY_DISTANCE_METERS:
                peer_safe = endpoint_peer_distance > current_peer_distance
            else:
                peer_safe = self.action_is_peer_safe(action)
            if peer_safe:
                moving_candidates.append(
                    (
                        -endpoint_peer_distance,
                        action,
                    )
                )

        if moving_candidates:
            best_score = min(
                candidate[0]
                for candidate in moving_candidates
            )
            best_actions = [
                candidate[1]
                for candidate in moving_candidates
                if candidate[0] == best_score
            ]
            left_actions = [
                action for action in best_actions if action[0] == "left"
            ]
            chosen = left_actions[0] if left_actions else best_actions[0]
            return (*chosen, "collision_avoidance_90_move")

        _, action_name, turn_degrees, next_heading = min(
            rotation_candidates
        )
        return (
            action_name,
            turn_degrees,
            next_heading,
            0.0,
            "collision_avoidance_90_turn_only",
        )

    def rotate_to_yaw(self, target_yaw):
        self.raise_if_failed()
        _, attitude = self.telemetry.snapshot()
        start_yaw = attitude[0]
        target_yaw = normalize_angle_degrees(target_yaw)
        deadline = time.monotonic() + self.motion_timeout(
            turn_degrees=normalize_angle_degrees(target_yaw - start_yaw)
        )

        while not self.stop_event.is_set():
            self.raise_if_failed()
            _, attitude = self.telemetry.snapshot()
            current_yaw = attitude[0]
            error = normalize_angle_degrees(target_yaw - current_yaw)
            if abs(error) <= TURN_TOLERANCE_DEGREES:
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("rotation timeout for {0}".format(self.name))

            turn_speed = min(self.args.turn_speed, max(MIN_TURN_SPEED_DPS, abs(error) * 0.8))
            self.drive_speed(
                x=0.0,
                y=0.0,
                z=math.copysign(turn_speed, error),
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)

        raise RuntimeError("rotation stopped before completion")

    def drive_forward(self, distance, target_yaw, collision_escape=False):
        self.raise_if_failed()
        position, attitude = self.telemetry.snapshot()
        start_x, start_y, _ = position
        target_yaw = normalize_angle_degrees(target_yaw)
        heading_radians = math.radians(target_yaw)
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        deadline = time.monotonic() + self.motion_timeout(distance=distance)
        initial_peer_distance = self.peer_distance(
            self.current_world_position()
        )
        boundary = self.start["partition"]

        while not self.stop_event.is_set():
            self.raise_if_failed()
            position, attitude = self.telemetry.snapshot()
            current_x, current_y, _ = position
            current_yaw = attitude[0]
            travelled = (current_x - start_x) * forward_x + (current_y - start_y) * forward_y
            remaining = distance - travelled
            if remaining <= DISTANCE_TOLERANCE_METERS:
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("forward timeout for {0}".format(self.name))

            world_position = self.current_world_position()
            peer_distance = self.peer_distance(world_position)
            if peer_distance < ROBOT_SAFETY_DISTANCE_METERS:
                moving_away = (
                    collision_escape
                    and peer_distance
                    >= initial_peer_distance - DISTANCE_TOLERANCE_METERS
                )
                if not moving_away:
                    self.command_stop()
                    print(
                        "{0}: forward motion stopped for robot collision "
                        "avoidance at {1:.3f} m.".format(
                            self.name,
                            peer_distance,
                        )
                    )
                    return "robot_guard"

            lookahead = project_point(
                world_position,
                self.current_world_heading(),
                min(BOUNDARY_LOOKAHEAD_METERS, max(remaining, 0.0)),
            )
            current_violation = boundary_violation(
                world_position,
                boundary,
            )
            lookahead_violation = boundary_violation(
                lookahead,
                boundary,
            )
            moving_farther_outside = (
                lookahead_violation
                > current_violation + 0.001
            )
            outside_odometry_tolerance = (
                lookahead_violation
                > BOUNDARY_ODOMETRY_TOLERANCE_METERS
            )
            if moving_farther_outside or outside_odometry_tolerance:
                self.command_stop()
                print(
                    "{0}: forward motion stopped at the arena boundary "
                    "(current violation={1:.3f} m, lookahead={2:.3f} m).".format(
                        self.name,
                        current_violation,
                        lookahead_violation,
                    )
                )
                return "boundary_guard"

            yaw_error = normalize_angle_degrees(target_yaw - current_yaw)
            forward_speed = min(self.args.linear_speed, max(MIN_LINEAR_SPEED_MPS, remaining * 0.8))
            self.drive_speed(
                x=forward_speed,
                y=0.0,
                z=0.0 if abs(yaw_error) <= HEADING_DRIFT_TOLERANCE_DEGREES else math.copysign(MIN_TURN_SPEED_DPS, yaw_error),
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)

        raise RuntimeError("forward motion stopped before completion")

    def valid_actions(self):
        current = self.current_world_position()
        partition = self.start["partition"]
        current_heading = self.current_world_heading()
        actions = []
        for action_name, turn_degrees in ACTION_TURNS.items():
            next_heading = normalize_angle_degrees(
                current_heading + turn_degrees
            )
            next_point = project_point(current, next_heading, self.args.step_distance)
            if inside_bounds(
                next_point,
                partition,
                tolerance=BOUNDARY_ODOMETRY_TOLERANCE_METERS,
            ):
                actions.append((action_name, turn_degrees, next_heading, self.args.step_distance))
        return actions

    def choose_boundary_recovery(self):
        """Turn by the shortest direction toward the arena center."""
        current = self.current_world_position()
        partition = self.start["partition"]
        center_x = 0.5 * (
            partition["x_min"] + partition["x_max"]
        )
        center_y = 0.5 * (
            partition["y_min"] + partition["y_max"]
        )
        inward_heading = math.degrees(
            math.atan2(
                center_y - current[1],
                center_x - current[0],
            )
        )
        current_heading = self.current_world_heading()
        turn_degrees = normalize_angle_degrees(
            inward_heading - current_heading
        )
        if abs(turn_degrees) <= TURN_TOLERANCE_DEGREES:
            turn_degrees = (
                ACTION_TURN_DEGREES
                if turn_degrees >= 0.0
                else -ACTION_TURN_DEGREES
            )
        action_name = "left" if turn_degrees > 0.0 else "right"
        next_heading = normalize_angle_degrees(
            current_heading + turn_degrees
        )
        return (
            action_name,
            turn_degrees,
            next_heading,
            0.0,
            "boundary_recovery_inward",
            {
                "selection_method": "turn_toward_arena_center",
                "candidate_actions": [action_name],
                "random_choice_verified": False,
            },
        )

    def choose_action(self):
        actions = self.valid_actions()
        if not actions:
            return self.choose_boundary_recovery()

        peer_safe_actions = [
            action for action in actions if self.action_is_peer_safe(action)
        ]
        if not peer_safe_actions:
            chosen = self.choose_collision_avoidance_action()
            return (
                *chosen,
                {
                    "selection_method": "safety_override",
                    "candidate_actions": ["left_90", "right_90"],
                    "random_choice_verified": False,
                }
            )
        actions = peer_safe_actions

        candidate_names = [action[0] for action in actions]
        chosen = self.random.choice(actions)
        return (
            *chosen,
            "random_walk",
            {
                "selection_method": "random.choice",
                "candidate_actions": candidate_names,
                "random_choice_verified": chosen in actions,
            }
        )

    def execute_random_step(self, step_index):
        self.raise_if_failed()
        (
            action_name,
            turn_degrees,
            next_heading,
            distance,
            decision_mode,
            selection_audit,
        ) = self.choose_action()
        action = {
            "step": step_index,
            "name": action_name,
            "turn_deg": round(turn_degrees, 3),
            "distance_m": round(distance, 4),
            "decision_mode": decision_mode,
            "selection_method": selection_audit["selection_method"],
            "candidate_actions": selection_audit["candidate_actions"],
            "random_choice_verified": selection_audit[
                "random_choice_verified"
            ],
            "chosen_utc": utc_timestamp(),
        }
        self.recorder.set_action(self.name, action)
        print(
            "{0}: step {1} action={2}, mode={3}, turn={4:+.1f} deg, "
            "distance={5:.3f} m, selection={6}, candidates={7}, "
            "verified={8}.".format(
                self.name,
                step_index,
                action_name,
                decision_mode,
                turn_degrees,
                distance,
                selection_audit["selection_method"],
                selection_audit["candidate_actions"],
                selection_audit["random_choice_verified"],
            )
        )

        self.command_stop()
        time.sleep(MOTION_SETTLE_SECONDS)
        self.raise_if_failed()
        self.rotate_to_yaw(self.target_sdk_yaw(next_heading))
        self.relative_heading = next_heading
        if distance > 0.0:
            collision_escape = (
                decision_mode == "collision_avoidance_90_move"
            )
            self.drive_forward(
                distance,
                self.target_sdk_yaw(self.relative_heading),
                collision_escape=collision_escape,
            )

    def run_random_walk(self):
        step_index = 0
        try:
            self.start_barrier.wait()
            deadline = time.monotonic() + self.args.duration
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self.raise_if_failed()
                self.execute_random_step(step_index)
                step_index += 1
            self.recorder.update_robot(self.name, completed=True)
        except ManualRobotFailure:
            self.recorder.update_robot(
                self.name,
                completed=True,
                failed=True,
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
        """Stop motion and close telemetry subscriptions."""
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
            "Run three RoboMaster robots with full-arena memoryless "
            "random-walk coverage. Press F in the terminal to turn robot 2 "
            "into a stationary failed obstacle."
        )
    )
    parser.add_argument("--linear-speed", type=float, default=DEFAULT_LINEAR_SPEED_MPS)
    parser.add_argument("--turn-speed", type=float, default=DEFAULT_TURN_SPEED_DPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--step-distance", type=float, default=DEFAULT_STEP_DISTANCE_METERS)
    parser.add_argument(
        "--world-yaw",
        type=float,
        default=None,
        help="Yaw angle, in SDK attitude degrees, to use as arena +x. Defaults to robot_1 startup yaw.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "random_walk_runs",
        help="Directory in which timestamped run folders are created",
    )
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()

    if not 0.08 <= args.linear_speed <= 1.0:
        raise SystemExit("--linear-speed must be between 0.08 and 1.0 m/s")
    if not 10.0 <= args.turn_speed <= 540.0:
        raise SystemExit("--turn-speed must be between 10 and 540 degrees/s")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be greater than zero")
    if args.step_distance <= 0.0:
        raise SystemExit("--step-distance must be greater than zero")
    return args


def main():
    # The SDK constructs a camera module for every robot during initialization.
    # Replace its LiveView before constructing Robot so no native H.264 or audio
    # decoder is created. This experiment records metadata only.
    media.LiveView = NoMediaLiveView

    args = parse_args()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = args.output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    starts = {
        name: build_robot_start(name, start_y)
        for name, _, start_y in ROBOT_SPECS
    }
    recorder = MetadataRecorder(run_directory / "metadata.json", starts, args)

    stop_event = threading.Event()
    start_barrier = threading.Barrier(len(ROBOT_SPECS) + 1)
    workers = []
    motion_threads = []

    print("Output: {0}".format(run_directory))
    print("Metadata-only mode: camera streams and image saving are disabled.")
    print(
        "Place robot centers at robot_1=({0:.4f}, {1:.4f}) m, "
        "robot_2=({0:.4f}, {2:.4f}) m, and "
        "robot_3=({0:.4f}, {3:.4f}) m.".format(
            starts["robot_1"]["start_center"]["x"],
            starts["robot_1"]["start_center"]["y"],
            starts["robot_2"]["start_center"]["y"],
            starts["robot_3"]["start_center"]["y"],
        )
    )
    print(
        "Point all three robots along +x. All robots may use the full "
        "{0:.1f} m by {1:.1f} m arena.".format(
            AREA_X_METERS,
            AREA_Y_METERS,
        )
    )
    print(
        "Collision avoidance uses a unique +/-{0:.0f} degree turn and "
        "forward escape move. Press F to fail robot_2; press Ctrl+C to "
        "stop the experiment.".format(
            COLLISION_AVOIDANCE_TURN_DEGREES
        )
    )

    try:
        for name, serial_number, _ in ROBOT_SPECS:
            worker = RobotWorker(
                name=name,
                serial_number=serial_number,
                start=starts[name],
                recorder=recorder,
                args=args,
                stop_event=stop_event,
                start_barrier=start_barrier,
            )
            worker.connect()
            workers.append(worker)

        world_yaw = args.world_yaw if args.world_yaw is not None else workers[0].initial_yaw
        if args.world_yaw is None:
            print("Using robot_1 startup yaw ({0:.2f} deg) as the shared arena +x.".format(world_yaw))
        else:
            print("Using --world-yaw ({0:.2f} deg) as the shared arena +x.".format(world_yaw))
        for worker in workers:
            worker.set_world_yaw(world_yaw)
            worker.set_peers(workers)

        recorder.record_positions(workers)
        for worker in workers:
            thread = threading.Thread(target=worker.run_random_walk, name="{0}-motion".format(worker.name))
            motion_threads.append(thread)
            thread.start()

        start_barrier.wait()
        motion_started = time.monotonic()
        failed_worker = next(
            worker for worker in workers
            if worker.name == FAILED_ROBOT_NAME
        )
        print("Motion started. Press F at any time to stop robot_2.")

        stop_sent = False
        next_metadata_sample = time.monotonic() + METADATA_SAMPLE_PERIOD_SECONDS
        while any(thread.is_alive() for thread in motion_threads):
            for thread in motion_threads:
                thread.join(timeout=0.02)
            now = time.monotonic()
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "f":
                    elapsed_seconds = now - motion_started
                    if not failed_worker.fail_as_obstacle(elapsed_seconds):
                        print(
                            "robot_2 is already a failed stationary obstacle "
                            "({0:.3f} s elapsed).".format(elapsed_seconds)
                        )
            if now >= next_metadata_sample:
                recorder.record_positions(workers)
                next_metadata_sample = now + METADATA_SAMPLE_PERIOD_SECONDS
            if stop_event.is_set() and not stop_sent:
                for worker in workers:
                    worker.stop_motion()
                stop_sent = True

        recorder.record_positions(workers)
        errors = [
            "{0}: {1}".format(name, recorder.errors[name])
            for name, _, _ in ROBOT_SPECS
            if recorder.errors[name]
        ]
        reason = (
            "Random-walk duration completed."
            if not errors and not stop_event.is_set()
            else "Random walk stopped before duration completed. {0}".format(" | ".join(errors))
        )
        recorder.finish(reason)
        print(reason)

    except KeyboardInterrupt:
        stop_event.set()
        for worker in workers:
            worker.stop_motion()
        if workers and all(worker.world_yaw is not None for worker in workers):
            recorder.record_positions(workers)
        try:
            start_barrier.abort()
        except threading.BrokenBarrierError:
            pass
        for thread in motion_threads:
            thread.join(timeout=3.0)
        recorder.finish("Interrupted by user.")
        print("Interrupted. All robots are stopping.")
    except Exception as error:
        stop_event.set()
        for worker in workers:
            worker.stop_motion()
        if workers and all(worker.world_yaw is not None for worker in workers):
            recorder.record_positions(workers)
        try:
            start_barrier.abort()
        except threading.BrokenBarrierError:
            pass
        recorder.finish("Run failed: {0}".format(error))
        print("Run failed: {0}".format(error))
    finally:
        stop_event.set()
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
