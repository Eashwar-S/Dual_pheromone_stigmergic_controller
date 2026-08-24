# -*-coding:utf-8-*-
"""Three-RoboMaster stigmergy search with a manually failed robot.

This keeps the robot setup, motion model, three-action camera-FOV action
space, boundary checks, and telemetry from random_walk.py.
Before every step, each robot divides its latest camera image into left,
straight, and right thirds and prefers the valid sector containing the least
significant pheromone-colored area. Each isolated camera-decoder process also
saves the unmodified BGR camera frames at 2 FPS in its robot's run folder.
Pressing F in the terminal abruptly stops robot 2, which remains a stationary
obstacle while all three camera streams continue saving images.
"""

import argparse
from collections import deque
from datetime import datetime
import math
import multiprocessing
import msvcrt
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

import random_walk as base
from robomaster import media


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


# EPCamera normally creates libmedia_codec decoders inside this process.
# Replace that LiveView before any robot is initialized. Each camera is
# decoded in its own subprocess below, so a native H.264 access violation
# cannot terminate motion control or metadata collection for all robots.
media.LiveView = ExternalDecoderLiveView


EDGE_MARGIN_METERS = 0.05
BOUNDARY_ODOMETRY_TOLERANCE_METERS = 0.02
BOUNDARY_LOOKAHEAD_METERS = 0.03
FORWARD_STALL_SECONDS = 2.0
FORWARD_PROGRESS_EPSILON_METERS = 0.005
ROBOT_DIAMETER_METERS = 2.0 * base.ROBOT_RADIUS_METERS
ROBOT_SAFETY_DISTANCE_METERS = ROBOT_DIAMETER_METERS
VISITED_CELL_SIZE_METERS = 0.20
COLLISION_AVOIDANCE_TURN_DEGREES = 90.0
INITIAL_ROBOT_SPACING_METERS = base.AREA_X_METERS / 3
ROBOT_SPECS = (
    ("robot_1", "3JKCH8800100VW", 0.0),
    ("robot_2", "3JKCH8800100RC", INITIAL_ROBOT_SPACING_METERS),
    ("robot_3", "3JKCH8800100VZ", 2 * INITIAL_ROBOT_SPACING_METERS),
)
FAILED_ROBOT_NAME = "robot_2"


# The two supplied HSV endpoints are normalized component-by-component so
# cv2.inRange always receives a valid lower and upper bound.
# PHEROMONE_HSV_ENDPOINT_A = (136, 80, 151)
# PHEROMONE_HSV_ENDPOINT_B = (128, 37, 162)
PHEROMONE_HSV_ENDPOINT_A = (140, 85, 174)
PHEROMONE_HSV_ENDPOINT_B = (110, 24, 148)
PHEROMONE_HSV_LOWER = np.array(
    [
        min(PHEROMONE_HSV_ENDPOINT_A[index], PHEROMONE_HSV_ENDPOINT_B[index])
        for index in range(3)
    ],
    dtype=np.uint8,
)
PHEROMONE_HSV_UPPER = np.array(
    [
        max(PHEROMONE_HSV_ENDPOINT_A[index], PHEROMONE_HSV_ENDPOINT_B[index])
        for index in range(3)
    ],
    dtype=np.uint8,
)
BLACK_HSV_LOWER = np.array((0, 0, 0), dtype=np.uint8)
BLACK_HSV_UPPER = np.array((179, 255, 55), dtype=np.uint8)
BLACK_NEAR_FIELD_TOP_FRACTION = 0.40
# The camera sees a narrow strip of the robot's own dark chassis at the
# bottom-center of every frame. Exclude that strip so it cannot combine with
# an out-of-arena side view and falsely report that every action is blocked.
BLACK_NEAR_FIELD_BOTTOM_FRACTION = 0.97

SECTOR_NAMES = ("left", "straight", "right")
CAMERA_FRAME_WAIT_SECONDS = 2.0
MAX_CAMERA_FRAME_AGE_SECONDS = 2.0
DEFAULT_PHEROMONE_MIN_PIXELS = 1000
DEFAULT_PHEROMONE_TIE_PIXELS = 200
DEFAULT_PHEROMONE_CONFIRM_FRAMES = 3
DEFAULT_BLACK_ROBOT_MIN_PIXELS = 1200
DEFAULT_CAPTURE_FPS = 2.0
DEFAULT_LINEAR_SPEED_MPS = 0.1


def build_robot_start(name, start_x):
    """Build a robot start inside the shared full-arena boundary."""
    return {
        "robot": name,
        "partition": {
            "x_min": 0.0,
            "x_max": base.AREA_X_METERS,
            "y_min": 0.0,
            "y_max": base.AREA_Y_METERS,
        },
        "start_center": base.round_point(
            (
                start_x + EDGE_MARGIN_METERS,
                EDGE_MARGIN_METERS,
            )
        ),
        "start_heading_deg": 0.0,
    }


def inside_world_bounds(point, partition, tolerance=0.0):
    """Check a robot-center point against the local safety margin."""
    return (
        distance_outside_world_bounds(point, partition)
        <= tolerance
    )


def distance_outside_world_bounds(point, partition):
    """Return Euclidean distance from a point to the safe inset rectangle."""
    x_min = partition["x_min"] + EDGE_MARGIN_METERS
    x_max = partition["x_max"] - EDGE_MARGIN_METERS
    y_min = partition["y_min"] + EDGE_MARGIN_METERS
    y_max = partition["y_max"] - EDGE_MARGIN_METERS
    x_error = max(x_min - point[0], 0.0, point[0] - x_max)
    y_error = max(y_min - point[1], 0.0, point[1] - y_max)
    return math.hypot(x_error, y_error)


class ManualRobotFailure(Exception):
    """End robot 2's motion without stopping cameras or the experiment."""


class MetadataRecorder(base.MetadataRecorder):
    """Extend the existing image-run metadata with manual failure state."""

    def __init__(self, path, starts, args):
        original_specs = base.ROBOT_SPECS
        base.ROBOT_SPECS = ROBOT_SPECS
        try:
            super().__init__(path, starts, args)
        finally:
            base.ROBOT_SPECS = original_specs

        self.failed = {name: False for name, _, _ in ROBOT_SPECS}
        with self.lock:
            self.data["manual_failure"] = {
                "robot": FAILED_ROBOT_NAME,
                "trigger_key": "f",
                "behavior": "abrupt_stop_stationary_obstacle",
            }
            self.data["failure_event"] = None
            self._write_locked()

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
        super().update_robot(name, completed=completed, error=error)
        if failed is not None:
            with self.lock:
                self.failed[name] = bool(failed)


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
    """Decode and save one robot's stream in an isolated native process."""
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
        lower = np.array(PHEROMONE_HSV_LOWER, dtype=np.uint8)
        upper = np.array(PHEROMONE_HSV_UPPER, dtype=np.uint8)
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
                # A temporary stream gap is normal on Wi-Fi. Keep this decoder
                # alive instead of reporting an error and exiting.
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

                now = time.monotonic()
                if now >= next_capture:
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
                                    "message": (
                                        "Could not save {0}"
                                    ).format(filename),
                                },
                            )
                    except Exception as error:
                        # Disk-writing trouble must not stop decoding or robot
                        # motion. Report it and continue using camera counts.
                        put_latest(
                            result_queue,
                            {
                                "kind": "warning",
                                "message": (
                                    "Image save failed: {0}: {1}"
                                ).format(type(error).__name__, error),
                            },
                        )

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                black_mask = cv2.inRange(hsv, black_lower, black_upper)
                boundaries = (0, width // 3, (2 * width) // 3, width)
                pheromone_counts = {
                    sector_name: int(
                        cv2.countNonZero(
                            mask[
                                :,
                                boundaries[index]:boundaries[index + 1],
                            ]
                        )
                    )
                    for index, sector_name in enumerate(SECTOR_NAMES)
                }
                near_field_top = int(
                    height * BLACK_NEAR_FIELD_TOP_FRACTION
                )
                near_field_bottom = int(
                    height * BLACK_NEAR_FIELD_BOTTOM_FRACTION
                )
                black_counts = {
                    sector_name: int(
                        cv2.countNonZero(
                            black_mask[
                                near_field_top:near_field_bottom,
                                boundaries[index]:boundaries[index + 1],
                            ]
                        )
                    )
                    for index, sector_name in enumerate(SECTOR_NAMES)
                }
                put_latest(
                    result_queue,
                    {
                        "kind": "counts",
                        "pheromone": pheromone_counts,
                        "black": black_counts,
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


class StigmergyRobotWorker(base.RobotWorker):
    """Robot worker that selects bounded actions from camera pheromone counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_lock = threading.Lock()
        self.camera_process_lock = threading.Lock()
        self.camera_context = multiprocessing.get_context("spawn")
        self.camera_result_queue = self.camera_context.Queue(maxsize=1)
        self.camera_process_stop_event = None
        self.camera_process = None
        self.camera_restart_count = 0
        self.recent_raw_counts = deque(
            maxlen=self.args.pheromone_confirm_frames
        )
        self.latest_counts_time = None
        self.frame_ready = threading.Event()
        self.last_steering_action = "straight"
        self.visited_lock = threading.Lock()
        self.visited_cells = set()
        self.peers = []
        self.failed_event = threading.Event()
        self.motion_command_lock = threading.RLock()

    def drive_speed(self, x, y, z, timeout=None):
        """Serialize commands and reject motion after manual failure."""
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
            timeout=base.DRIVE_COMMAND_TIMEOUT_SECONDS,
        )

    def raise_if_failed(self):
        if self.failed_event.is_set():
            raise ManualRobotFailure()

    def fail_as_obstacle(self, elapsed_seconds):
        """Abruptly stop robot 2 while leaving its camera recorder active."""
        with self.motion_command_lock:
            if self.failed_event.is_set():
                return False
            self.failed_event.set()
            self.chassis.drive_speed(
                x=0.0,
                y=0.0,
                z=0.0,
                timeout=base.DRIVE_COMMAND_TIMEOUT_SECONDS,
            )

        action = {
            "name": "failed_obstacle",
            "turn_deg": 0.0,
            "distance_m": 0.0,
            "trigger": "terminal_key_f",
            "elapsed_seconds": round(elapsed_seconds, 3),
            "failed_utc": base.utc_timestamp(),
        }
        self.recorder.record_manual_failure(
            self.name,
            action,
            elapsed_seconds,
        )
        print(
            "\n{0}: F pressed after {1:.3f} s. Motion stopped abruptly; "
            "the robot remains a stationary obstacle and its camera keeps "
            "saving images.".format(self.name, elapsed_seconds)
        )
        return True

    def set_peers(self, workers):
        self.peers = [worker for worker in workers if worker is not self]

    def position_cell(self, point):
        return (
            int(math.floor(point[0] / VISITED_CELL_SIZE_METERS)),
            int(math.floor(point[1] / VISITED_CELL_SIZE_METERS)),
        )

    def mark_visited(self, point=None):
        if point is None:
            point = self.current_world_position()
        with self.visited_lock:
            self.visited_cells.add(self.position_cell(point))

    def action_endpoint(self, action):
        return base.project_point(
            self.current_world_position(),
            action[2],
            action[3],
        )

    def action_is_unvisited(self, action):
        current = self.current_world_position()
        current_cell = self.position_cell(current)
        sample_spacing = VISITED_CELL_SIZE_METERS / 2.0
        sample_count = max(
            1,
            int(math.ceil(action[3] / sample_spacing)),
        )
        path_cells = {
            self.position_cell(
                base.project_point(
                    current,
                    action[2],
                    action[3] * sample_index / sample_count,
                )
            )
            for sample_index in range(1, sample_count + 1)
        }
        path_cells.discard(current_cell)
        with self.visited_lock:
            return not path_cells.intersection(self.visited_cells)

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
            separation = math.hypot(
                closest_x - peer_point[0],
                closest_y - peer_point[1],
            )
            if separation < ROBOT_SAFETY_DISTANCE_METERS:
                return False
        return True

    def connect(self):
        super().connect()
        self.start_camera_decoder()

    def start_camera_decoder(self):
        with self.camera_process_lock:
            if self.camera_process is not None and self.camera_process.is_alive():
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
                self.recent_raw_counts.clear()
                self.latest_counts_time = None
                self.frame_ready.clear()

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
        """Receive HSV counts; native frames remain in the decoder process."""
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

            if result.get("kind") == "error":
                print(
                    "{0}: camera decoder warning: {1}".format(
                        self.name,
                        result.get("message"),
                    )
                )
                continue

            if result.get("kind") == "warning":
                print(
                    "{0}: camera warning: {1}".format(
                        self.name,
                        result.get("message"),
                    )
                )
                continue

            if result.get("kind") == "counts":
                with self.camera_lock:
                    self.recent_raw_counts.append(
                        {
                            "pheromone": result["pheromone"],
                            "black": result["black"],
                        }
                    )
                    self.latest_counts_time = time.monotonic()
                    enough_frames = (
                        len(self.recent_raw_counts)
                        >= self.args.pheromone_confirm_frames
                    )
                if enough_frames:
                    self.frame_ready.set()

    def pheromone_counts(self):
        """Return median-filtered and thresholded counts for all sectors."""
        if not self.frame_ready.wait(timeout=CAMERA_FRAME_WAIT_SECONDS):
            return None

        with self.camera_lock:
            samples = list(self.recent_raw_counts)
            counts_time = self.latest_counts_time
        if (
            len(samples) < self.args.pheromone_confirm_frames
            or counts_time is None
            or time.monotonic() - counts_time > MAX_CAMERA_FRAME_AGE_SECONDS
        ):
            return None

        raw_counts = {
            sector_name: int(
                np.median(
                    [
                        sample["pheromone"][sector_name]
                        for sample in samples
                    ]
                )
            )
            for sector_name in SECTOR_NAMES
        }
        effective_counts = {
            sector_name: (
                count
                if count >= self.args.pheromone_min_pixels
                else 0
            )
            for sector_name, count in raw_counts.items()
        }
        return {
            "raw": raw_counts,
            "effective": effective_counts,
            "black": {
                sector_name: int(
                    np.median(
                        [
                            sample["black"][sector_name]
                            for sample in samples
                        ]
                    )
                )
                for sector_name in SECTOR_NAMES
            },
        }

    def choose_least_marked_action(self, actions, effective_counts):
        """Prefer straight when pheromone counts are effectively tied."""
        minimum_count = min(effective_counts[action[0]] for action in actions)
        near_minimum = [
            action
            for action in actions
            if effective_counts[action[0]]
            <= minimum_count + self.args.pheromone_tie_pixels
        ]
        minimum_turn_cost = min(abs(action[1]) for action in near_minimum)
        best_actions = [
            action
            for action in near_minimum
            if abs(action[1]) == minimum_turn_cost
        ]
        previous_matches = [
            action
            for action in best_actions
            if action[0] == self.last_steering_action
        ]
        if previous_matches:
            return previous_matches[0]
        return self.random.choice(best_actions)

    def choose_boundary_recovery_action(self):
        """Turn consistently toward the center of the shared arena."""
        current_x, current_y, _ = self.current_world_position()
        partition = self.start["partition"]
        center_x = (partition["x_min"] + partition["x_max"]) / 2.0
        center_y = (partition["y_min"] + partition["y_max"]) / 2.0
        center_heading = math.degrees(
            math.atan2(center_y - current_y, center_x - current_x)
        )

        candidates = []
        for action_name in ("left", "right"):
            turn_degrees = base.ACTION_TURNS[action_name]
            next_heading = base.normalize_angle_degrees(
                self.relative_heading + turn_degrees
            )
            heading_error = abs(
                base.normalize_angle_degrees(center_heading - next_heading)
            )
            candidates.append(
                (heading_error, action_name, turn_degrees, next_heading)
            )
        _, action_name, turn_degrees, next_heading = min(candidates)
        return action_name, turn_degrees, next_heading

    def choose_rotation_from_actions(self, actions, sector_scores=None):
        """Choose a non-translating turn, preferring the lowest sector score."""
        turning_actions = [
            action for action in actions if action[0] in ("left", "right")
        ]
        if not turning_actions:
            action_name, turn_degrees, next_heading = (
                self.choose_boundary_recovery_action()
            )
            return action_name, turn_degrees, next_heading

        if sector_scores is not None:
            minimum_score = min(
                sector_scores[action[0]] for action in turning_actions
            )
            turning_actions = [
                action
                for action in turning_actions
                if sector_scores[action[0]] == minimum_score
            ]

        previous_matches = [
            action
            for action in turning_actions
            if action[0] == self.last_steering_action
        ]
        chosen = (
            previous_matches[0]
            if previous_matches
            else self.random.choice(turning_actions)
        )
        return chosen[0], chosen[1], chosen[2]

    def choose_collision_avoidance_action(self, sector_scores=None):
        """Choose a unique 90-degree turn-and-forward collision escape."""
        current = self.current_world_position()
        partition = self.start["partition"]
        current_peer_distance = self.peer_distance(current)
        moving_candidates = []
        rotation_candidates = []

        for action_name, turn_degrees in (
            ("left", COLLISION_AVOIDANCE_TURN_DEGREES),
            ("right", -COLLISION_AVOIDANCE_TURN_DEGREES),
        ):
            next_heading = base.normalize_angle_degrees(
                self.relative_heading + turn_degrees
            )
            endpoint = base.project_point(
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
            black_score = (
                0
                if sector_scores is None
                else sector_scores[action_name]
            )
            endpoint_peer_distance = self.peer_distance(endpoint)
            rotation_candidates.append(
                (
                    black_score,
                    -endpoint_peer_distance,
                    action_name,
                    turn_degrees,
                    next_heading,
                )
            )

            if not inside_world_bounds(
                endpoint,
                partition,
                tolerance=BOUNDARY_ODOMETRY_TOLERANCE_METERS,
            ):
                continue

            if current_peer_distance < ROBOT_SAFETY_DISTANCE_METERS:
                peer_safe = (
                    endpoint_peer_distance
                    > current_peer_distance
                    + FORWARD_PROGRESS_EPSILON_METERS
                )
            else:
                peer_safe = self.action_is_peer_safe(action)
            if peer_safe:
                moving_candidates.append(
                    (
                        black_score,
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
            return (
                left_actions[0] if left_actions else best_actions[0],
                "collision_avoidance_90_move",
            )

        # If neither full movement is safe, still make the distinct 90-degree
        # turn that points toward the better escape side, then reassess.
        _, _, action_name, turn_degrees, next_heading = min(
            rotation_candidates
        )
        return (
            (action_name, turn_degrees, next_heading, 0.0),
            "collision_avoidance_90_turn_only",
        )

    def valid_actions(self):
        """Filter projected actions using the failure-run boundary rules."""
        current = self.current_world_position()
        partition = self.start["partition"]
        actions = []
        for action_name, turn_degrees in base.ACTION_TURNS.items():
            next_heading = base.normalize_angle_degrees(
                self.relative_heading + turn_degrees
            )
            next_point = base.project_point(
                current,
                next_heading,
                self.args.step_distance,
            )
            if inside_world_bounds(
                next_point,
                partition,
                tolerance=BOUNDARY_ODOMETRY_TOLERANCE_METERS,
            ):
                actions.append(
                    (
                        action_name,
                        turn_degrees,
                        next_heading,
                        self.args.step_distance,
                    )
                )
        return actions

    def rotate_to_yaw(self, target_yaw):
        """Rotate without the inherited post-action settle delay."""
        self.raise_if_failed()
        _, attitude = self.telemetry.snapshot()
        start_yaw = attitude[0]
        target_yaw = base.normalize_angle_degrees(target_yaw)
        deadline = time.monotonic() + self.motion_timeout(
            turn_degrees=base.normalize_angle_degrees(
                target_yaw - start_yaw
            )
        )

        while not self.stop_event.is_set():
            self.raise_if_failed()
            _, attitude = self.telemetry.snapshot()
            current_yaw = attitude[0]
            error = base.normalize_angle_degrees(target_yaw - current_yaw)
            if abs(error) <= base.TURN_TOLERANCE_DEGREES:
                self.command_stop()
                return "completed"
            if time.monotonic() >= deadline:
                self.command_stop()
                print(
                    "{0}: rotation timed out; continuing with the next "
                    "decision.".format(self.name)
                )
                return "timeout"

            turn_speed = min(
                self.args.turn_speed,
                max(base.MIN_TURN_SPEED_DPS, abs(error) * 0.8),
            )
            self.drive_speed(
                x=0.0,
                y=0.0,
                z=math.copysign(turn_speed, error),
                timeout=base.DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(base.CONTROL_PERIOD_SECONDS)

        self.command_stop()
        return "stopped"

    def drive_forward(
        self,
        distance,
        target_yaw,
        collision_escape=False,
    ):
        """Drive with live boundary and stall guards, without settle delay."""
        self.raise_if_failed()
        position, _ = self.telemetry.snapshot()
        start_x, start_y, _ = position
        target_yaw = base.normalize_angle_degrees(target_yaw)
        heading_radians = math.radians(target_yaw)
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        deadline = time.monotonic() + self.motion_timeout(distance=distance)
        last_progress = 0.0
        last_progress_time = time.monotonic()
        partition = self.start["partition"]
        initial_peer_distance = self.peer_distance(
            self.current_world_position()
        )

        while not self.stop_event.is_set():
            self.raise_if_failed()
            position, attitude = self.telemetry.snapshot()
            current_x, current_y, _ = position
            current_yaw = attitude[0]
            travelled = (
                (current_x - start_x) * forward_x
                + (current_y - start_y) * forward_y
            )
            remaining = distance - travelled

            if remaining <= base.DISTANCE_TOLERANCE_METERS:
                self.command_stop()
                return "completed"

            world_position = self.current_world_position()
            self.mark_visited(world_position)
            peer_distance = self.peer_distance(world_position)
            if peer_distance < ROBOT_SAFETY_DISTANCE_METERS:
                moving_away = (
                    collision_escape
                    and peer_distance
                    >= initial_peer_distance
                    - FORWARD_PROGRESS_EPSILON_METERS
                )
                if not moving_away:
                    self.command_stop()
                    print(
                        "{0}: stopped because another robot is {1:.3f} m "
                        "away.".format(self.name, peer_distance)
                    )
                    return "robot_guard"

            with self.camera_lock:
                camera_samples = list(self.recent_raw_counts)
                camera_time = self.latest_counts_time
            if (
                camera_samples
                and camera_time is not None
                and time.monotonic() - camera_time
                <= MAX_CAMERA_FRAME_AGE_SECONDS
            ):
                forward_black_pixels = int(
                    np.median(
                        [
                            sample["black"]["straight"]
                            for sample in camera_samples
                        ]
                    )
                )
                if (
                    forward_black_pixels
                    >= self.args.black_robot_min_pixels
                ):
                    self.command_stop()
                    print(
                        "{0}: stopped for a near-field black object "
                        "({1} pixels).".format(
                            self.name,
                            forward_black_pixels,
                        )
                    )
                    return "black_robot_guard"

            lookahead_point = base.project_point(
                world_position,
                self.relative_heading,
                min(BOUNDARY_LOOKAHEAD_METERS, max(remaining, 0.0)),
            )
            current_boundary_error = distance_outside_world_bounds(
                world_position,
                partition,
            )
            lookahead_boundary_error = distance_outside_world_bounds(
                lookahead_point,
                partition,
            )
            moving_farther_outside = (
                lookahead_boundary_error
                > current_boundary_error + 0.001
            )
            outside_odometry_tolerance = (
                lookahead_boundary_error
                > BOUNDARY_ODOMETRY_TOLERANCE_METERS
            )
            if moving_farther_outside or outside_odometry_tolerance:
                self.command_stop()
                print(
                    "{0}: forward motion stopped at the {1:.2f} m world "
                    "margin.".format(self.name, EDGE_MARGIN_METERS)
                )
                return "boundary_guard"

            now = time.monotonic()
            if travelled >= last_progress + FORWARD_PROGRESS_EPSILON_METERS:
                last_progress = travelled
                last_progress_time = now
            elif now - last_progress_time >= FORWARD_STALL_SECONDS:
                self.command_stop()
                print(
                    "{0}: forward motion stalled after {1:.3f} m; "
                    "continuing with a new decision.".format(
                        self.name,
                        max(0.0, travelled),
                    )
                )
                return "stalled"

            if now >= deadline:
                self.command_stop()
                print(
                    "{0}: forward motion timed out after {1:.3f} m; "
                    "continuing with a new decision.".format(
                        self.name,
                        max(0.0, travelled),
                    )
                )
                return "timeout"

            yaw_error = base.normalize_angle_degrees(
                target_yaw - current_yaw
            )
            forward_speed = min(
                self.args.linear_speed,
                max(base.MIN_LINEAR_SPEED_MPS, remaining * 0.8),
            )
            self.drive_speed(
                x=forward_speed,
                y=0.0,
                z=(
                    0.0
                    if abs(yaw_error)
                    <= base.HEADING_DRIFT_TOLERANCE_DEGREES
                    else math.copysign(
                        base.MIN_TURN_SPEED_DPS,
                        yaw_error,
                    )
                ),
                timeout=base.DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(base.CONTROL_PERIOD_SECONDS)

        self.command_stop()
        return "stopped"

    def choose_stigmergy_action(self):
        """Choose an unvisited, robot-safe, least-pheromone sector."""
        self.mark_visited()
        actions = self.valid_actions()
        pheromone = self.pheromone_counts()

        if not actions:
            (
                action_name,
                turn_degrees,
                next_heading,
            ) = self.choose_boundary_recovery_action()
            self.last_steering_action = action_name
            return (
                action_name,
                turn_degrees,
                next_heading,
                0.0,
                pheromone,
                "boundary_recovery",
            )

        peer_safe_actions = [
            action for action in actions if self.action_is_peer_safe(action)
        ]
        if not peer_safe_actions:
            chosen, decision_mode = (
                self.choose_collision_avoidance_action()
            )
            self.last_steering_action = chosen[0]
            return (*chosen, pheromone, decision_mode)
        actions = peer_safe_actions

        if pheromone is not None:
            # Do not remove a telemetry-safe action solely because its broad
            # camera sector contains black pixels. Near an arena edge, the FOV
            # can include dark floor outside the boundary even when the
            # projected robot path turns safely inward. Peer telemetry and
            # action_is_peer_safe already protect against all active and failed
            # robots; the live straight-ahead black guard remains active while
            # driving.
            effective_counts = pheromone["effective"]
            if all(
                effective_counts[action[0]] > 0
                for action in actions
            ):
                # Even when every safe direction contains pheromone, retain
                # the stigmergy rule and take the least-marked safe sector.
                # Random selection here previously allowed a heavily marked
                # straight sector to win over a much clearer turn.
                chosen = self.choose_least_marked_action(
                    actions,
                    effective_counts,
                )
                self.last_steering_action = chosen[0]
                return (
                    *chosen,
                    pheromone,
                    "least_pheromone_surrounded",
                )

            # Pheromone is the primary navigation signal. Restrict exploration
            # and visited-cell tie-breaking to sectors at the minimum effective
            # count; otherwise an unvisited but heavily marked path could beat
            # the genuinely least-marked action.
            minimum_count = min(
                effective_counts[action[0]]
                for action in actions
            )
            actions = [
                action
                for action in actions
                if effective_counts[action[0]]
                <= minimum_count + self.args.pheromone_tie_pixels
            ]

        unvisited_actions = [
            action for action in actions if self.action_is_unvisited(action)
        ]
        if not unvisited_actions:
            # All safe paths cross visited cells. Continue through the
            # least-marked safe path instead of allowing a random recovery
            # to override the pheromone measurement.
            if pheromone is None:
                chosen = self.random.choice(actions)
                decision_mode = "random_visited_recovery_no_frame"
            else:
                chosen = self.choose_least_marked_action(
                    actions,
                    pheromone["effective"],
                )
                decision_mode = "least_pheromone_visited_recovery"
            self.last_steering_action = chosen[0]
            return (
                *chosen,
                pheromone,
                decision_mode,
            )
        actions = unvisited_actions

        if pheromone is None:
            straight_actions = [
                action for action in actions if action[0] == "straight"
            ]
            chosen = (
                straight_actions[0]
                if straight_actions
                else self.choose_least_marked_action(
                    actions,
                    {action[0]: 0 for action in actions},
                )
            )
            self.last_steering_action = chosen[0]
            return (*chosen, None, "no_frame_safe_path")

        effective_counts = pheromone["effective"]
        chosen = self.choose_least_marked_action(actions, effective_counts)
        self.last_steering_action = chosen[0]
        decision_mode = (
            "clear_path"
            if all(effective_counts[action[0]] == 0 for action in actions)
            else "least_pheromone"
        )
        return (*chosen, pheromone, decision_mode)

    def execute_stigmergy_step(self, step_index):
        self.raise_if_failed()
        (
            action_name,
            turn_degrees,
            next_heading,
            distance,
            pheromone,
            decision_mode,
        ) = self.choose_stigmergy_action()
        raw_counts = None if pheromone is None else pheromone["raw"]
        effective_counts = (
            None if pheromone is None else pheromone["effective"]
        )
        action = {
            "step": step_index,
            "name": action_name,
            "turn_deg": round(turn_degrees, 3),
            "distance_m": round(distance, 4),
            "decision_mode": decision_mode,
            "pheromone_pixels": raw_counts,
            "pheromone_effective_pixels": effective_counts,
            "pheromone_min_pixels": self.args.pheromone_min_pixels,
            "black_robot_pixels": (
                None if pheromone is None else pheromone["black"]
            ),
            "black_robot_min_pixels": self.args.black_robot_min_pixels,
            "visited_cells": len(self.visited_cells),
            "chosen_utc": base.utc_timestamp(),
        }
        self.recorder.set_action(self.name, action)
        print(
            "{0}: step {1} action={2}, mode={3}, pheromone={4}, "
            "turn={5:+.1f} deg, distance={6:.3f} m.".format(
                self.name,
                step_index,
                action_name,
                decision_mode,
                effective_counts,
                turn_degrees,
                distance,
            )
        )

        rotation_result = self.rotate_to_yaw(
            self.target_sdk_yaw(next_heading)
        )
        if rotation_result == "completed":
            self.relative_heading = next_heading
        else:
            self.relative_heading = self.current_world_heading()
        drive_result = "not_requested"
        if distance > 0.0 and rotation_result == "completed":
            collision_escape = (
                decision_mode == "collision_avoidance_90_move"
            )
            if collision_escape:
                # Counts collected before the 90-degree turn describe the old
                # camera directions. Discard them so they cannot immediately
                # cancel the escape movement as stale black-object evidence.
                with self.camera_lock:
                    self.recent_raw_counts.clear()
                    self.latest_counts_time = None
                    self.frame_ready.clear()
            drive_result = self.drive_forward(
                distance,
                self.target_sdk_yaw(self.relative_heading),
                collision_escape=collision_escape,
            )
        elif distance > 0.0:
            drive_result = "skipped_after_rotation_{0}".format(
                rotation_result
            )
        action["rotation_result"] = rotation_result
        action["drive_result"] = drive_result
        self.recorder.set_action(self.name, action)

    def run_stigmergy_search(self):
        step_index = 0
        deadline = time.monotonic() + self.args.duration
        try:
            self.start_barrier.wait()
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self.raise_if_failed()
                self.execute_stigmergy_step(step_index)
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

    def close(self):
        """Stop isolated decoding before shutting down the robot stream."""
        self.stop_event.set()
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
        super().close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run three bounded RoboMaster stigmergy-search robots that avoid "
            "camera-visible pheromone. Press F to fail robot 2."
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
        default=base.DEFAULT_TURN_SPEED_DPS,
    )
    parser.add_argument(
        "--pheromone-min-pixels",
        type=int,
        default=DEFAULT_PHEROMONE_MIN_PIXELS,
        help=(
            "Minimum HSV-matching pixels required for a sector to count as "
            "containing pheromone"
        ),
    )
    parser.add_argument(
        "--pheromone-tie-pixels",
        type=int,
        default=DEFAULT_PHEROMONE_TIE_PIXELS,
        help=(
            "Treat sector counts within this many pixels as tied, preferring "
            "the action with the smallest turn"
        ),
    )
    parser.add_argument(
        "--pheromone-confirm-frames",
        type=int,
        default=DEFAULT_PHEROMONE_CONFIRM_FRAMES,
        help=(
            "Median-filter this many recent camera frames before making "
            "a pheromone decision"
        ),
    )
    parser.add_argument(
        "--black-robot-min-pixels",
        type=int,
        default=DEFAULT_BLACK_ROBOT_MIN_PIXELS,
        help=(
            "Minimum near-field black pixels in a camera sector before it "
            "is treated as another robot"
        ),
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_CAPTURE_FPS,
        help="Raw JPEG capture rate per robot (default: 2 FPS)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=base.DEFAULT_DURATION_SECONDS,
    )
    parser.add_argument(
        "--step-distance",
        type=float,
        default=base.DEFAULT_STEP_DISTANCE_METERS,
    )
    parser.add_argument(
        "--world-yaw",
        type=float,
        default=None,
        help=(
            "Yaw angle, in SDK attitude degrees, to use as arena +x. "
            "Defaults to robot_1 startup yaw."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "stigmergy_search_runs",
        help="Directory in which timestamped run folders are created",
    )
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()

    if not 0.08 <= args.linear_speed <= 1.0:
        raise SystemExit("--linear-speed must be between 0.08 and 1.0 m/s")
    if not 10.0 <= args.turn_speed <= 540.0:
        raise SystemExit("--turn-speed must be between 10 and 540 degrees/s")
    if args.pheromone_min_pixels < 1:
        raise SystemExit("--pheromone-min-pixels must be at least 1")
    if args.pheromone_tie_pixels < 0:
        raise SystemExit("--pheromone-tie-pixels cannot be negative")
    if args.pheromone_confirm_frames < 1:
        raise SystemExit("--pheromone-confirm-frames must be at least 1")
    if args.black_robot_min_pixels < 1:
        raise SystemExit("--black-robot-min-pixels must be at least 1")
    if args.capture_fps <= 0.0:
        raise SystemExit("--capture-fps must be greater than zero")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be greater than zero")
    if args.step_distance <= 0.0:
        raise SystemExit("--step-distance must be greater than zero")
    return args


def main():
    args = parse_args()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = args.output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    starts = {
        name: build_robot_start(name, start_x)
        for name, _, start_x in ROBOT_SPECS
    }
    recorder = MetadataRecorder(
        run_directory / "metadata.json",
        starts,
        args,
    )

    stop_event = threading.Event()
    start_barrier = threading.Barrier(len(ROBOT_SPECS) + 1)
    workers = []
    motion_threads = []

    print("Output: {0}".format(run_directory))
    print(
        "Place robot centers at robot_1=({0:.4f}, {1:.4f}) m, "
        "robot_2=({2:.4f}, {1:.4f}) m, and "
        "robot_3=({3:.4f}, {1:.4f}) m.".format(
            starts["robot_1"]["start_center"]["x"],
            starts["robot_1"]["start_center"]["y"],
            starts["robot_2"]["start_center"]["x"],
            starts["robot_3"]["start_center"]["x"],
        )
    )
    print(
        "Point all three robots along +x. HSV pheromone range: {0} to {1}.".format(
            tuple(int(value) for value in PHEROMONE_HSV_LOWER),
            tuple(int(value) for value in PHEROMONE_HSV_UPPER),
        )
    )
    print(
        "World boundary margin: {0:.2f} m. Post-action settle delays are "
        "disabled.".format(EDGE_MARGIN_METERS)
    )
    print(
        "Each robot avoids its own visited {0:.2f} m cells. Robot avoidance "
        "uses a {1:.3f} m telemetry distance and {2} near-field black "
        "pixels.".format(
            VISITED_CELL_SIZE_METERS,
            ROBOT_SAFETY_DISTANCE_METERS,
            args.black_robot_min_pixels,
        )
    )
    print(
        "Collision avoidance uses a unique +/-{0:.0f} degree turn followed "
        "by a {1:.2f} m escape move when the route is safe.".format(
            COLLISION_AVOIDANCE_TURN_DEGREES,
            args.step_distance,
        )
    )
    print(
        "A sector needs at least {0} matching pixels to count as pheromone; "
        "counts within {1} pixels are treated as tied. Decisions use the "
        "median of {2} frames.".format(
            args.pheromone_min_pixels,
            args.pheromone_tie_pixels,
            args.pheromone_confirm_frames,
        )
    )
    print(
        "Unmodified camera images are saved at {0:.1f} FPS in the robot_1, "
        "robot_2, and robot_3 folders. Each stream uses an isolated "
        "restartable H.264 decoder process.".format(args.capture_fps)
    )
    print("Press F to fail robot_2; press Ctrl+C to stop all robots.")

    try:
        for name, serial_number, _ in ROBOT_SPECS:
            image_directory = run_directory / name
            image_directory.mkdir()
            worker = StigmergyRobotWorker(
                name=name,
                serial_number=serial_number,
                start=starts[name],
                image_directory=image_directory,
                recorder=recorder,
                args=args,
                stop_event=stop_event,
                start_barrier=start_barrier,
            )
            workers.append(worker)
            worker.connect()

        world_yaw = (
            args.world_yaw
            if args.world_yaw is not None
            else workers[0].initial_yaw
        )
        if args.world_yaw is None:
            print(
                "Using robot_1 startup yaw ({0:.2f} deg) as the shared "
                "arena +x.".format(world_yaw)
            )
        else:
            print(
                "Using --world-yaw ({0:.2f} deg) as the shared arena +x.".format(
                    world_yaw
                )
            )
        for worker in workers:
            worker.set_world_yaw(world_yaw)
            worker.set_peers(workers)

        recorder.record_positions(workers)
        for worker in workers:
            thread = threading.Thread(
                target=worker.run_stigmergy_search,
                name="{0}-motion".format(worker.name),
            )
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
        next_metadata_sample = (
            time.monotonic() + base.METADATA_SAMPLE_PERIOD_SECONDS
        )
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
                next_metadata_sample = (
                    now + base.METADATA_SAMPLE_PERIOD_SECONDS
                )
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
            "Stigmergy-search duration completed."
            if not errors and not stop_event.is_set()
            else "Stigmergy search stopped before duration completed. {0}".format(
                " | ".join(errors)
            )
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
