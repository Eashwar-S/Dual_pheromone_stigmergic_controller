# -*-coding:utf-8-*-
"""Memoryless two-RoboMaster random-walk coverage with collision avoidance.

Both robots share the entire arena. At each normal decision, each robot
randomly chooses one camera-FOV sector: left, straight, or right. Choices that
would leave the arena are filtered out. Robot collisions use a distinct
90-degree turn-and-forward escape maneuver; no pheromone logic is used.
"""

import argparse
from datetime import datetime
import json
import math
import multiprocessing
from pathlib import Path
import queue
import random
import sys
import threading
import time

import cv2
import numpy as np


# Prefer this SDK checkout over an older robomaster package in site-packages.
SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from robomaster import media, robot


class ExternalDecoderLiveView:
    """Keep native media decoders out of the motion-control process."""

    def __init__(self, ep_robot):
        self._robot = ep_robot

    def start_video_stream(self, display=False, addr=None, ip_proto="tcp"):
        return True

    def stop_video_stream(self):
        return True

    def stop(self):
        pass


# main() installs this replacement only when random_walk.py runs directly.
# Keeping the assignment out of module import preserves camera behavior for
# other scripts that import RobotWorker and constants from this file.


AREA_X_METERS = 4.0
AREA_Y_METERS = 1.8
ROBOT_COUNT = 2
INITIAL_ROBOT_SPACING_METERS = AREA_X_METERS / ROBOT_COUNT
# Retained for older scripts that import this constant; random_walk.py no
# longer uses it as a movement partition.
PARTITION_WIDTH_METERS = INITIAL_ROBOT_SPACING_METERS
ROBOT_SPECS = (
    ("robot_1", "3JKCH8800100VW", 0.0),
    ("robot_2", "3JKCH8800100RC", INITIAL_ROBOT_SPACING_METERS),
)
ROBOT_RADIUS_METERS = 8.0 * 0.0254
EDGE_MARGIN_METERS = 0.10
ROBOT_SAFETY_DISTANCE_METERS = 2.0 * ROBOT_RADIUS_METERS
COLLISION_AVOIDANCE_TURN_DEGREES = 90.0
BLACK_HSV_LOWER = np.array((0, 0, 0), dtype=np.uint8)
BLACK_HSV_UPPER = np.array((179, 255, 55), dtype=np.uint8)
BLACK_NEAR_FIELD_TOP_FRACTION = 0.40
BLACK_ROBOT_MIN_PIXELS = 1200
MAX_CAMERA_FRAME_AGE_SECONDS = 2.0
BOUNDARY_LOOKAHEAD_METERS = 0.03

DEFAULT_LINEAR_SPEED_MPS = 0.1
DEFAULT_TURN_SPEED_DPS = 30.0
DEFAULT_CAPTURE_FPS = 5.0
DEFAULT_DURATION_SECONDS = 300.0
DEFAULT_STEP_DISTANCE_METERS = 0.25
CAMERA_HORIZONTAL_FOV_DEGREES = 90.0
ACTION_TURN_DEGREES = CAMERA_HORIZONTAL_FOV_DEGREES / 3.0
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
    "straight": 0.0,
    "right": -ACTION_TURN_DEGREES,
}


def put_latest(result_queue, value):
    """Replace an old camera result instead of allowing queue backlog."""
    try:
        result_queue.put_nowait(value)
        return
    except queue.Full:
        pass

    try:
        result_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        result_queue.put_nowait(value)
    except queue.Full:
        pass


def camera_decoder_process(
    name,
    stream_address,
    stream_protocol,
    result_queue,
    process_stop_event,
    image_directory,
    capture_fps,
):
    """Decode, analyze, and save one camera in an isolated process."""
    from robomaster import conn
    import libmedia_codec

    stream_connection = conn.StreamConnection()
    try:
        if not stream_connection.connect(stream_address, stream_protocol):
            put_latest(
                result_queue,
                {
                    "kind": "error",
                    "message": "Could not connect to camera stream",
                },
            )
            return

        decoder = libmedia_codec.H264Decoder()
        black_lower = np.array(BLACK_HSV_LOWER, dtype=np.uint8)
        black_upper = np.array(BLACK_HSV_UPPER, dtype=np.uint8)
        image_directory = Path(image_directory)
        image_directory.mkdir(parents=True, exist_ok=True)
        capture_period = 1.0 / capture_fps
        next_capture = time.monotonic()
        frame_index = 0

        while not process_stop_event.is_set():
            try:
                encoded_data = stream_connection.read_buf(timeout=0.5)
            except queue.Empty:
                continue
            if not encoded_data:
                continue

            frames = decoder.decode(encoded_data)
            for frame_data in frames:
                frame_bytes, width, height, line_size = frame_data
                if not frame_bytes:
                    continue

                frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                black_mask = cv2.inRange(hsv, black_lower, black_upper)
                boundaries = (0, width // 3, (2 * width) // 3, width)
                near_field_top = int(
                    height * BLACK_NEAR_FIELD_TOP_FRACTION
                )
                black_counts = {
                    sector_name: int(
                        cv2.countNonZero(
                            black_mask[
                                near_field_top:,
                                boundaries[index]:boundaries[index + 1],
                            ]
                        )
                    )
                    for index, sector_name in enumerate(
                        ("left", "straight", "right")
                    )
                }
                put_latest(
                    result_queue,
                    {
                        "kind": "counts",
                        "black": black_counts,
                    },
                )

                now = time.monotonic()
                if now < next_capture:
                    continue
                next_capture = now + capture_period
                frame_index += 1
                filename = "frame_{0:06d}_{1}.jpg".format(
                    frame_index,
                    datetime.utcnow().strftime(
                        "%Y%m%dT%H%M%S_%f"
                    )[:-3],
                )
                try:
                    if not cv2.imwrite(
                        str(image_directory / filename),
                        frame,
                    ):
                        put_latest(
                            result_queue,
                            {
                                "kind": "warning",
                                "message": "Could not save {0}".format(
                                    filename
                                ),
                            },
                        )
                except Exception as error:
                    put_latest(
                        result_queue,
                        {
                            "kind": "warning",
                            "message": "Image save failed: {0}: {1}".format(
                                type(error).__name__,
                                error,
                            ),
                        },
                    )
    except Exception as error:
        put_latest(
            result_queue,
            {
                "kind": "error",
                "message": "{0}: {1}".format(type(error).__name__, error),
            },
        )
    finally:
        try:
            stream_connection.disconnect()
        except Exception:
            pass


def utc_timestamp():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def round_point(point):
    return {"x": round(point[0], 4), "y": round(point[1], 4)}


def normalize_angle_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def build_robot_start(name, start_x):
    x_min = start_x + EDGE_MARGIN_METERS
    y_min = EDGE_MARGIN_METERS
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


def inside_bounds(point, partition):
    return (
        partition["x_min"] + EDGE_MARGIN_METERS <= point[0] <= partition["x_max"] - EDGE_MARGIN_METERS
        and partition["y_min"] + EDGE_MARGIN_METERS <= point[1] <= partition["y_max"] - EDGE_MARGIN_METERS
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
        self.coverage = CoverageTracker(COVERAGE_CELL_SIZE_METERS, ROBOT_RADIUS_METERS)
        self.data = {
            "schema_version": 1,
            "run_started_utc": utc_timestamp(),
            "run_finished_utc": None,
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

    def update_robot(self, name, completed=None, error=None):
        with self.lock:
            if completed is not None:
                self.completed[name] = bool(completed)
            if error is not None:
                self.errors[name] = error

    def record_positions(self, workers):
        timestamp = utc_timestamp()
        sample = {"timestamp_utc": timestamp, "robots": {}}

        with self.lock:
            actions = dict(self.current_actions)

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
    USE_ISOLATED_CAMERA_DECODER = True

    def __init__(
        self,
        name,
        serial_number,
        start,
        image_directory,
        recorder,
        args,
        stop_event,
        start_barrier,
    ):
        self.name = name
        self.serial_number = serial_number
        self.start = start
        self.image_directory = image_directory
        self.recorder = recorder
        self.args = args
        self.stop_event = stop_event
        self.start_barrier = start_barrier
        self.ep_robot = robot.Robot()
        self.chassis = None
        self.ep_camera = None
        self.initialized = False
        self.video_started = False
        self.position_subscribed = False
        self.attitude_subscribed = False
        self.capture_thread = None
        self.frame_index = 0
        self.telemetry = MotionTelemetry()
        self.initial_yaw = None
        self.initial_position = None
        self.world_yaw = None
        self.relative_heading = 0.0
        self.random = random.Random(args.seed + sum(ord(char) for char in name))
        self.camera_lock = threading.Lock()
        self.latest_black_counts = None
        self.latest_camera_time = None
        self.camera_process_lock = threading.Lock()
        self.camera_context = multiprocessing.get_context("spawn")
        self.camera_result_queue = self.camera_context.Queue(maxsize=1)
        self.camera_process_stop_event = None
        self.camera_process = None
        self.camera_restart_count = 0
        self.peers = []

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

        self.ep_camera = self.ep_robot.camera
        self.ep_camera.start_video_stream(
            display=False,
        )
        self.video_started = True
        if self.USE_ISOLATED_CAMERA_DECODER:
            self.start_camera_decoder()

        self.capture_thread = threading.Thread(
            target=self.capture_frames,
            name="{0}-camera".format(self.name),
            daemon=True,
        )
        self.capture_thread.start()

    def set_world_yaw(self, world_yaw):
        self.world_yaw = normalize_angle_degrees(world_yaw)
        print("{0}: arena +x yaw target is {1:.2f} deg.".format(self.name, self.world_yaw))

    def start_camera_decoder(self):
        with self.camera_process_lock:
            if (
                self.camera_process is not None
                and self.camera_process.is_alive()
            ):
                return

            if self.camera_process is not None:
                self.camera_process.join(timeout=0.2)
                self.camera_process.close()

            while True:
                try:
                    self.camera_result_queue.get_nowait()
                except queue.Empty:
                    break
            with self.camera_lock:
                self.latest_black_counts = None
                self.latest_camera_time = None

            self.camera_process_stop_event = self.camera_context.Event()
            self.camera_process = self.camera_context.Process(
                target=camera_decoder_process,
                args=(
                    self.name,
                    self.ep_camera.video_stream_addr,
                    self.ep_camera.conf.video_stream_proto,
                    self.camera_result_queue,
                    self.camera_process_stop_event,
                    str(self.image_directory),
                    self.args.capture_fps,
                ),
                name="{0}-h264-decoder".format(self.name),
                daemon=True,
            )
            self.camera_process.start()
            self.camera_restart_count += 1
            print(
                "{0}: isolated H.264 decoder started (instance {1}).".format(
                    self.name,
                    self.camera_restart_count,
                )
            )

    def capture_frames(self):
        """Receive camera counts and restart a failed native decoder."""
        while not self.stop_event.is_set():
            try:
                result = self.camera_result_queue.get(timeout=0.5)
            except queue.Empty:
                if (
                    self.camera_process is not None
                    and not self.camera_process.is_alive()
                    and not self.stop_event.is_set()
                ):
                    print(
                        "{0}: H.264 decoder exited with code {1}; "
                        "restarting it.".format(
                            self.name,
                            self.camera_process.exitcode,
                        )
                    )
                    self.start_camera_decoder()
                continue

            if result.get("kind") in ("error", "warning"):
                print(
                    "{0}: camera decoder warning: {1}".format(
                        self.name,
                        result.get("message"),
                    )
                )
                continue

            if result.get("kind") == "counts":
                with self.camera_lock:
                    self.latest_black_counts = result["black"]
                    self.latest_camera_time = time.monotonic()

    def motion_timeout(self, distance=0.0, turn_degrees=0.0):
        expected_seconds = distance / self.args.linear_speed + abs(turn_degrees) / self.args.turn_speed
        return expected_seconds * MOTION_TIMEOUT_MULTIPLIER + MOTION_TIMEOUT_MARGIN_SECONDS

    def command_stop(self):
        self.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=DRIVE_COMMAND_TIMEOUT_SECONDS)

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

    def current_black_counts(self):
        with self.camera_lock:
            counts = self.latest_black_counts
            timestamp = self.latest_camera_time
        if (
            counts is None
            or timestamp is None
            or time.monotonic() - timestamp > MAX_CAMERA_FRAME_AGE_SECONDS
        ):
            return None
        return dict(counts)

    def choose_collision_avoidance_action(self, black_counts=None):
        """Choose a boundary-safe 90-degree collision escape."""
        current = self.current_world_position()
        boundary = self.start["partition"]
        current_peer_distance = self.peer_distance(current)
        moving_candidates = []
        rotation_candidates = []

        for action_name, turn_degrees in (
            ("left", COLLISION_AVOIDANCE_TURN_DEGREES),
            ("right", -COLLISION_AVOIDANCE_TURN_DEGREES),
        ):
            next_heading = normalize_angle_degrees(
                self.relative_heading + turn_degrees
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
            visual_score = (
                0 if black_counts is None else black_counts[action_name]
            )
            endpoint_peer_distance = self.peer_distance(endpoint)
            rotation_candidates.append(
                (
                    visual_score,
                    -endpoint_peer_distance,
                    action_name,
                    turn_degrees,
                    next_heading,
                )
            )

            if not inside_bounds(endpoint, boundary):
                continue
            if current_peer_distance < ROBOT_SAFETY_DISTANCE_METERS:
                peer_safe = endpoint_peer_distance > current_peer_distance
            else:
                peer_safe = self.action_is_peer_safe(action)
            if peer_safe:
                moving_candidates.append(
                    (
                        visual_score,
                        -endpoint_peer_distance,
                        action,
                    )
                )

        if moving_candidates:
            best_score = min(
                (candidate[0], candidate[1])
                for candidate in moving_candidates
            )
            best_actions = [
                candidate[2]
                for candidate in moving_candidates
                if (candidate[0], candidate[1]) == best_score
            ]
            left_actions = [
                action for action in best_actions if action[0] == "left"
            ]
            chosen = left_actions[0] if left_actions else best_actions[0]
            return (*chosen, "collision_avoidance_90_move")

        _, _, action_name, turn_degrees, next_heading = min(
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
        _, attitude = self.telemetry.snapshot()
        start_yaw = attitude[0]
        target_yaw = normalize_angle_degrees(target_yaw)
        deadline = time.monotonic() + self.motion_timeout(
            turn_degrees=normalize_angle_degrees(target_yaw - start_yaw)
        )

        while not self.stop_event.is_set():
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
            self.chassis.drive_speed(
                x=0.0,
                y=0.0,
                z=math.copysign(turn_speed, error),
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)

        raise RuntimeError("rotation stopped before completion")

    def drive_forward(self, distance, target_yaw, collision_escape=False):
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

            black_counts = self.current_black_counts()
            if (
                black_counts is not None
                and black_counts["straight"] >= BLACK_ROBOT_MIN_PIXELS
            ):
                self.command_stop()
                print(
                    "{0}: forward motion stopped for a near-field robot "
                    "({1} black pixels).".format(
                        self.name,
                        black_counts["straight"],
                    )
                )
                return "black_robot_guard"

            lookahead = project_point(
                world_position,
                self.relative_heading,
                min(BOUNDARY_LOOKAHEAD_METERS, max(remaining, 0.0)),
            )
            if not inside_bounds(lookahead, boundary):
                self.command_stop()
                print(
                    "{0}: forward motion stopped at the arena boundary.".format(
                        self.name
                    )
                )
                return "boundary_guard"

            yaw_error = normalize_angle_degrees(target_yaw - current_yaw)
            forward_speed = min(self.args.linear_speed, max(MIN_LINEAR_SPEED_MPS, remaining * 0.8))
            self.chassis.drive_speed(
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
        actions = []
        for action_name, turn_degrees in ACTION_TURNS.items():
            next_heading = normalize_angle_degrees(self.relative_heading + turn_degrees)
            next_point = project_point(current, next_heading, self.args.step_distance)
            if inside_bounds(next_point, partition):
                actions.append((action_name, turn_degrees, next_heading, self.args.step_distance))
        return actions

    def choose_action(self):
        actions = self.valid_actions()
        if not actions:
            # At a corner, rotate randomly without translating until a
            # forward sector becomes valid again.
            action_name = self.random.choice(("left", "right"))
            turn_degrees = ACTION_TURNS[action_name]
            next_heading = normalize_angle_degrees(
                self.relative_heading + turn_degrees
            )
            return (
                action_name,
                turn_degrees,
                next_heading,
                0.0,
                "boundary_recovery",
                {
                    "selection_method": "random.choice",
                    "candidate_actions": ["left", "right"],
                    "random_choice_verified": True,
                },
            )

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

        black_counts = self.current_black_counts()
        if black_counts is not None:
            black_safe_actions = [
                action
                for action in actions
                if black_counts[action[0]] < BLACK_ROBOT_MIN_PIXELS
            ]
            if not black_safe_actions:
                chosen = self.choose_collision_avoidance_action(black_counts)
                return (
                    *chosen,
                    {
                        "selection_method": "safety_override",
                        "candidate_actions": ["left_90", "right_90"],
                        "random_choice_verified": False,
                    }
                )
            actions = black_safe_actions

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
        self.rotate_to_yaw(self.target_sdk_yaw(next_heading))
        self.relative_heading = next_heading
        if distance > 0.0:
            collision_escape = (
                decision_mode == "collision_avoidance_90_move"
            )
            if collision_escape:
                with self.camera_lock:
                    self.latest_black_counts = None
                    self.latest_camera_time = None
            self.drive_forward(
                distance,
                self.target_sdk_yaw(self.relative_heading),
                collision_escape=collision_escape,
            )

    def run_random_walk(self):
        step_index = 0
        deadline = time.monotonic() + self.args.duration
        try:
            self.start_barrier.wait()
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self.execute_random_step(step_index)
                step_index += 1
            self.recorder.update_robot(self.name, completed=True)
        except Exception as error:
            print("{0} failed: {1}".format(self.name, error))
            self.recorder.update_robot(self.name, error=str(error))
            self.stop_event.set()
        finally:
            self.stop_motion()

    def stop_motion(self):
        if self.chassis is not None:
            try:
                self.chassis.drive_speed(x=0.0, y=0.0, z=0.0)
            except Exception:
                pass

    def close(self):
        """Stop isolated decoding before shutting down the robot stream."""
        self.stop_motion()
        if self.camera_process_stop_event is not None:
            self.camera_process_stop_event.set()
        if self.camera_process is not None:
            self.camera_process.join(timeout=4.0)
            if self.camera_process.is_alive():
                self.camera_process.terminate()
                self.camera_process.join(timeout=2.0)
            self.camera_process.close()
            self.camera_process = None
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
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
        if self.video_started and self.ep_camera is not None:
            try:
                self.ep_camera.stop_video_stream()
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
            "Run two RoboMaster robots with full-arena memoryless random-walk "
            "coverage and collision avoidance."
        )
    )
    parser.add_argument("--linear-speed", type=float, default=DEFAULT_LINEAR_SPEED_MPS)
    parser.add_argument("--turn-speed", type=float, default=DEFAULT_TURN_SPEED_DPS)
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_CAPTURE_FPS,
        help="JPEG capture rate per robot",
    )
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
    if args.capture_fps <= 0.0:
        raise SystemExit("--capture-fps must be greater than zero")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be greater than zero")
    if args.step_distance <= 0.0:
        raise SystemExit("--step-distance must be greater than zero")
    return args


def main():
    # Each camera is decoded in its own subprocess. A native Windows H.264
    # crash can therefore be restarted without terminating robot motion or
    # metadata. Install this before either RobotWorker constructs robot.Robot.
    media.LiveView = ExternalDecoderLiveView

    args = parse_args()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = args.output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    starts = {
        name: build_robot_start(name, start_x)
        for name, _, start_x in ROBOT_SPECS
    }
    recorder = MetadataRecorder(run_directory / "metadata.json", starts, args)

    stop_event = threading.Event()
    start_barrier = threading.Barrier(len(ROBOT_SPECS))
    workers = []
    motion_threads = []

    print("Output: {0}".format(run_directory))
    print(
        "Place robot_1 center at ({0:.4f}, {1:.4f}) m and robot_2 center at ({2:.4f}, {1:.4f}) m.".format(
            starts["robot_1"]["start_center"]["x"],
            starts["robot_1"]["start_center"]["y"],
            starts["robot_2"]["start_center"]["x"],
        )
    )
    print(
        "Point both robots along +x. Both robots may use the full "
        "{0:.1f} m by {1:.1f} m arena.".format(
            AREA_X_METERS,
            AREA_Y_METERS,
        )
    )
    print(
        "Collision avoidance uses a unique +/-{0:.0f} degree turn and "
        "forward escape move. Press Ctrl+C to stop.".format(
            COLLISION_AVOIDANCE_TURN_DEGREES
        )
    )

    try:
        for name, serial_number, _ in ROBOT_SPECS:
            image_directory = run_directory / name
            image_directory.mkdir()
            worker = RobotWorker(
                name=name,
                serial_number=serial_number,
                start=starts[name],
                image_directory=image_directory,
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

        stop_sent = False
        next_metadata_sample = time.monotonic() + METADATA_SAMPLE_PERIOD_SECONDS
        while any(thread.is_alive() for thread in motion_threads):
            for thread in motion_threads:
                thread.join(timeout=0.1)
            now = time.monotonic()
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
        print("Interrupted. Both robots are stopping.")
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
