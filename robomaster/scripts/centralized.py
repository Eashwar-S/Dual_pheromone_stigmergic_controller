# -*-coding:utf-8-*-
"""Centralized two-RoboMaster lawnmower coverage with metadata collection."""

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time

# Prefer this SDK checkout over an older robomaster package in site-packages.
SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from robomaster import robot


class MetadataOnlyLiveView:
    """Prevent the SDK's unused camera module from creating media decoders."""

    def __init__(self, ep_robot):
        self._robot = ep_robot

    def stop(self):
        pass


# Robot.initialize() always constructs an EPCamera, whose normal constructor
# creates H.264 and Opus decoder objects even when no stream is requested.
# Substitute a decoder-free LiveView before either Robot is initialized.
robot.camera.media.LiveView = MetadataOnlyLiveView


AREA_X_METERS = 4.0
AREA_Y_METERS = 1.8
ROBOT_COUNT = 2
PARTITION_WIDTH_METERS = AREA_X_METERS / ROBOT_COUNT
ROBOT_SPECS = (
    ("robot_1", "3JKCH8800100VW", 0.0),
    ("robot_2", "3JKCH8800100RC", PARTITION_WIDTH_METERS),
)
ROBOT_RADIUS_METERS = 10.0 * 0.0254
ROBOT_DIAMETER_METERS = ROBOT_RADIUS_METERS * 2.0
EDGE_MARGIN_METERS = 0.0

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


def utc_timestamp():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def round_point(point):
    return {"x": round(point[0], 4), "y": round(point[1], 4)}


def normalize_angle_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def make_lane_positions():
    """Choose lanes whose circular swaths cover y=[0, AREA_Y_METERS]."""
    minimum = EDGE_MARGIN_METERS
    maximum = AREA_Y_METERS - EDGE_MARGIN_METERS
    span = maximum - minimum
    if span <= 0.0:
        return [AREA_Y_METERS / 2.0]
    interval_count = max(1, int(math.ceil(span / ROBOT_DIAMETER_METERS)))
    spacing = span / interval_count
    return [minimum + index * spacing for index in range(interval_count + 1)]


def build_robot_plan(name, partition_x_min):
    """Build world-axis moves with yaw aligned to each travel direction."""
    x_min = partition_x_min + EDGE_MARGIN_METERS
    x_max = partition_x_min + PARTITION_WIDTH_METERS - EDGE_MARGIN_METERS
    lanes = make_lane_positions()
    actions = []
    world_x = x_min
    world_y = lanes[0]
    heading = 0.0

    for row_index, lane_y in enumerate(lanes):
        destination_x = x_max if row_index % 2 == 0 else x_min
        start = (world_x, world_y)
        end = (destination_x, lane_y)
        row_distance = abs(destination_x - world_x)
        actions.append(
            {
                "kind": "traverse",
                "row": row_index,
                "command": {
                    "x_m": round(row_distance, 4),
                    "y_m": 0.0,
                    "z_deg": 0.0,
                },
                "world_start": round_point(start),
                "world_end": round_point(end),
                "heading_start_deg": heading,
                "heading_end_deg": heading,
            }
        )
        world_x, world_y = end

        if row_index == len(lanes) - 1:
            continue

        next_lane_y = lanes[row_index + 1]
        lane_shift = next_lane_y - lane_y
        turn_degrees = 90.0 if row_index % 2 == 0 else -90.0

        actions.append(
            {
                "kind": "turn_toward_next_lane",
                "row": row_index,
                "command": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_deg": turn_degrees,
                },
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((world_x, world_y)),
                "heading_start_deg": heading,
                "heading_end_deg": heading + turn_degrees,
            }
        )
        heading += turn_degrees

        actions.append(
            {
                "kind": "lane_shift",
                "row": row_index,
                "command": {
                    "x_m": round(lane_shift, 4),
                    "y_m": 0.0,
                    "z_deg": 0.0,
                },
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((world_x, next_lane_y)),
                "heading_start_deg": heading,
                "heading_end_deg": heading,
            }
        )
        world_y = next_lane_y

        actions.append(
            {
                "kind": "turn_onto_next_row",
                "row": row_index,
                "command": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_deg": turn_degrees,
                },
                "world_start": round_point((world_x, world_y)),
                "world_end": round_point((world_x, world_y)),
                "heading_start_deg": heading,
                "heading_end_deg": heading + turn_degrees,
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
        "lane_spacing_m": (
            round(lanes[1] - lanes[0], 4)
            if len(lanes) > 1
            else 0.0
        ),
        "actions": actions,
    }


class MetadataRecorder:
    def __init__(self, path, plans, args):
        self.path = path
        self.lock = threading.Lock()
        self.current_actions = {
            name: {"step": None, "kind": "idle"}
            for name, _, _ in ROBOT_SPECS
        }
        self.completed_rows = {name: 0 for name, _, _ in ROBOT_SPECS}
        self.completed = {name: False for name, _, _ in ROBOT_SPECS}
        self.errors = {name: None for name, _, _ in ROBOT_SPECS}
        self.runtime = {
            name: {"connected": False}
            for name, _, _ in ROBOT_SPECS
        }
        self.data = {
            "schema_version": 2,
            "run_started_utc": utc_timestamp(),
            "run_finished_utc": None,
            "world_start": {
                name: {
                    "x": plan["start_center"]["x"],
                    "y": plan["start_center"]["y"],
                    "heading_deg": plan["start_heading_deg"],
                }
                for name, plan in plans.items()
            },
            "samples": [],
            "coverage": {
                "complete": False,
                "end_reason": None,
            },
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
            self.runtime[name].update(values)
            if "completed" in values:
                self.completed[name] = bool(values["completed"])
            if "error" in values:
                self.errors[name] = values["error"]

    def begin_action(self, name, action_record):
        with self.lock:
            self.current_actions[name] = {
                "step": action_record["step"],
                "kind": action_record["kind"],
            }
            self._write_locked()
            return action_record["step"]

    def finish_action(self, name, action_index, values):
        with self.lock:
            if values.get("status") == "completed":
                current_action = self.current_actions[name]
                if current_action["kind"] == "traverse":
                    self.completed_rows[name] += 1
                self.current_actions[name] = {
                    "step": current_action["step"],
                    "kind": "idle",
                }
            elif values.get("status") == "failed":
                self.current_actions[name] = {
                    "step": action_index,
                    "kind": "failed",
                }
            self._write_locked()

    def record_positions(self, workers):
        timestamp = utc_timestamp()
        sample = {"timestamp_utc": timestamp, "robots": {}}
        with self.lock:
            actions = copy.deepcopy(self.current_actions)

        for worker in workers:
            position = worker.current_world_position()
            sample["robots"][worker.name] = {
                "position": {
                    "x": round(position[0], 4),
                    "y": round(position[1], 4),
                    "z": round(position[2], 4),
                },
                "action": actions[worker.name],
            }

        with self.lock:
            self.data["samples"].append(sample)
            self._write_locked()

    def finish(self, complete, reason, lane_count):
        with self.lock:
            self.data["coverage"].update(
                {
                    "complete": complete,
                    "end_reason": reason,
                }
            )
            self.data["run_finished_utc"] = utc_timestamp()
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
        plan,
        recorder,
        args,
        stop_event,
        start_barrier,
    ):
        self.name = name
        self.serial_number = serial_number
        self.plan = plan
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

    def connect(self):
        initialized = bool(
            self.ep_robot.initialize(
                conn_type="sta",
                sn=self.serial_number,
            )
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
                "Could not subscribe to motion telemetry for {0}".format(
                    self.name
                )
            )
        if not self.telemetry.wait_until_ready(timeout=5.0):
            raise RuntimeError(
                "Timed out waiting for motion telemetry for {0}".format(
                    self.name
                )
            )
        position, attitude = self.telemetry.snapshot()
        self.initial_position = position
        self.initial_yaw = attitude[0]
        self.recorder.update_robot(
            self.name,
            initial_yaw_deg=round(self.initial_yaw, 3),
            initial_position=position,
        )
        print(
            "{0}: startup yaw is {1:.2f} deg.".format(
                self.name,
                self.initial_yaw,
            )
        )

        self.recorder.update_robot(self.name, connected=True)

    def set_world_yaw(self, world_yaw):
        self.world_yaw = normalize_angle_degrees(world_yaw)
        print(
            "{0}: arena +x yaw target is {1:.2f} deg.".format(
                self.name,
                self.world_yaw,
            )
        )

    def motion_timeout(self, distance=0.0, turn_degrees=0.0):
        expected_seconds = (
            distance / self.args.linear_speed
            + abs(turn_degrees) / self.args.turn_speed
        )
        return (
            expected_seconds * MOTION_TIMEOUT_MULTIPLIER
            + MOTION_TIMEOUT_MARGIN_SECONDS
        )

    def command_stop(self):
        self.chassis.drive_speed(
            x=0.0,
            y=0.0,
            z=0.0,
            timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
        )

    def rotate_to_yaw(self, target_yaw):
        _, attitude = self.telemetry.snapshot()
        start_yaw = attitude[0]
        target_yaw = normalize_angle_degrees(target_yaw)
        requested_turn = normalize_angle_degrees(target_yaw - start_yaw)
        deadline = time.monotonic() + self.motion_timeout(
            turn_degrees=requested_turn
        )

        while not self.stop_event.is_set():
            _, attitude = self.telemetry.snapshot()
            current_yaw = attitude[0]
            error = normalize_angle_degrees(target_yaw - current_yaw)
            if abs(error) <= TURN_TOLERANCE_DEGREES:
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                return {
                    "start_yaw_deg": round(start_yaw, 3),
                    "target_yaw_deg": round(target_yaw, 3),
                    "final_yaw_deg": round(current_yaw, 3),
                    "yaw_error_deg": round(error, 3),
                }
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "rotation timeout: target={0:.1f}, current={1:.1f}, "
                    "error={2:.1f}".format(target_yaw, current_yaw, error)
                )

            turn_speed = min(
                self.args.turn_speed,
                max(MIN_TURN_SPEED_DPS, abs(error) * 0.8),
            )
            self.chassis.drive_speed(
                x=0.0,
                y=0.0,
                z=math.copysign(turn_speed, error),
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)

        raise RuntimeError("rotation stopped before completion")

    def rotate_in_place(self, turn_degrees):
        _, attitude = self.telemetry.snapshot()
        target_yaw = attitude[0] + turn_degrees
        return self.rotate_to_yaw(target_yaw)

    def planned_yaw(self, relative_heading):
        if self.world_yaw is None:
            raise RuntimeError("World yaw was not set for {0}".format(self.name))
        return normalize_angle_degrees(self.world_yaw + relative_heading)

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

    def drive_forward(self, distance, target_yaw):
        position, attitude = self.telemetry.snapshot()
        start_x, start_y, _ = position
        target_yaw = normalize_angle_degrees(target_yaw)
        heading_radians = math.radians(target_yaw)
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        right_x = -forward_y
        right_y = forward_x
        heading_corrections = 0
        last_correction_travelled = 0.0
        deadline = time.monotonic() + self.motion_timeout(distance=distance)

        while not self.stop_event.is_set():
            position, attitude = self.telemetry.snapshot()
            current_x, current_y, _ = position
            current_yaw = attitude[0]
            delta_x = current_x - start_x
            delta_y = current_y - start_y
            travelled = (
                delta_x * forward_x
                + delta_y * forward_y
            )
            cross_track = delta_x * right_x + delta_y * right_y
            remaining = distance - travelled
            if remaining <= DISTANCE_TOLERANCE_METERS:
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                return {
                    "requested_distance_m": round(distance, 4),
                    "travelled_distance_m": round(travelled, 4),
                    "cross_track_error_m": round(cross_track, 4),
                    "target_yaw_deg": round(target_yaw, 3),
                    "final_yaw_deg": round(current_yaw, 3),
                    "heading_corrections": heading_corrections,
                    "position_start": round_point((start_x, start_y)),
                    "position_end": round_point((current_x, current_y)),
                }
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "forward timeout: requested={0:.3f} m, "
                    "travelled={1:.3f} m".format(distance, travelled)
                )

            yaw_error = normalize_angle_degrees(target_yaw - current_yaw)
            correction_spacing = travelled - last_correction_travelled
            if (
                abs(yaw_error) > HEADING_DRIFT_TOLERANCE_DEGREES
                and correction_spacing >= MIN_DISTANCE_BETWEEN_HEADING_CORRECTIONS_METERS
            ):
                self.command_stop()
                time.sleep(MOTION_SETTLE_SECONDS)
                self.rotate_to_yaw(target_yaw)
                heading_corrections += 1
                last_correction_travelled = travelled
                continue

            forward_speed = min(
                self.args.linear_speed,
                max(MIN_LINEAR_SPEED_MPS, remaining * 0.8),
            )
            self.chassis.drive_speed(
                x=forward_speed,
                y=0.0,
                z=0.0,
                timeout=DRIVE_COMMAND_TIMEOUT_SECONDS,
            )
            time.sleep(CONTROL_PERIOD_SECONDS)

        raise RuntimeError("forward motion stopped before completion")

    def execute_action(self, step_index, planned_action):
        command = planned_action["command"]
        record = copy.deepcopy(planned_action)
        record.update(
            {
                "step": step_index,
                "chosen_utc": utc_timestamp(),
                "started_utc": None,
                "finished_utc": None,
                "status": "chosen",
                "sdk_state": None,
            }
        )
        action_index = self.recorder.begin_action(self.name, record)

        is_turn = abs(command["z_deg"]) > 0.0
        distance = math.hypot(command["x_m"], command["y_m"])
        timeout = self.motion_timeout(
            distance=distance,
            turn_degrees=command["z_deg"],
        )
        print(
            "{0}: step {1} {2}, command={3}, heading {4:.1f}->{5:.1f}, "
            "timeout={6:.1f}s".format(
                self.name,
                step_index,
                planned_action["kind"],
                command,
                planned_action["heading_start_deg"],
                planned_action["heading_end_deg"],
                timeout,
            )
        )
        started_utc = utc_timestamp()
        try:
            self.command_stop()
            time.sleep(MOTION_SETTLE_SECONDS)
            if is_turn:
                target_yaw = self.planned_yaw(
                    planned_action["heading_end_deg"]
                )
                measured = self.rotate_to_yaw(target_yaw)
                sdk_state = "yaw_target_reached"
            else:
                target_yaw = self.planned_yaw(
                    planned_action["heading_start_deg"]
                )
                _, attitude = self.telemetry.snapshot()
                initial_heading_error = normalize_angle_degrees(
                    target_yaw - attitude[0]
                )
                pre_drive_alignment = None
                if abs(initial_heading_error) > TURN_TOLERANCE_DEGREES:
                    pre_drive_alignment = self.rotate_to_yaw(target_yaw)
                measured = self.drive_forward(distance, target_yaw)
                measured["pre_drive_alignment"] = pre_drive_alignment
                sdk_state = "distance_target_reached"
            status = "completed"
            self.recorder.finish_action(
                self.name,
                action_index,
                {
                    "started_utc": started_utc,
                    "finished_utc": utc_timestamp(),
                    "sdk_state": sdk_state,
                    "measured": measured,
                    "status": status,
                },
            )
        except Exception as error:
            self.command_stop()
            self.recorder.finish_action(
                self.name,
                action_index,
                {
                    "started_utc": started_utc,
                    "finished_utc": utc_timestamp(),
                    "sdk_state": "control_error",
                    "error": str(error),
                    "status": "failed",
                },
            )
            raise

        print(
            "{0}: step {1} {2} completed.".format(
                self.name,
                step_index,
                planned_action["kind"],
            )
        )

    def run_plan(self):
        try:
            self.start_barrier.wait()
            for step_index, planned_action in enumerate(self.plan["actions"]):
                if self.stop_event.is_set():
                    raise RuntimeError("Stopped because another worker failed")
                self.execute_action(step_index, planned_action)

            self.recorder.update_robot(self.name, completed=True)
        except Exception as error:
            print("{0} failed: {1}".format(self.name, error))
            self.recorder.update_robot(self.name, error=str(error))
            self.stop_event.set()
        finally:
            if self.chassis is not None:
                try:
                    self.chassis.drive_speed(x=0.0, y=0.0, z=0.0)
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

    def stop_motion(self):
        if self.chassis is not None:
            try:
                self.chassis.drive_speed(x=0.0, y=0.0, z=0.0)
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run two RoboMaster robots through equal lawnmower partitions "
            "using separate yaw turns and forward-only translations."
        )
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=DEFAULT_LINEAR_SPEED_MPS,
        help="Maximum forward drive speed in m/s",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=DEFAULT_TURN_SPEED_DPS,
        help="Turn speed in degrees/s",
    )
    parser.add_argument(
        "--world-yaw",
        type=float,
        default=None,
        help=(
            "Yaw angle, in SDK attitude degrees, to use as arena +x. "
            "Defaults to each robot's own startup yaw."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "centralized_runs",
        help="Directory in which timestamped run folders are created",
    )
    args = parser.parse_args()

    if not 0.08 <= args.linear_speed <= 1.0:
        raise SystemExit("--linear-speed must be between 0.08 and 1.0 m/s")
    if not 10.0 <= args.turn_speed <= 540.0:
        raise SystemExit("--turn-speed must be between 10 and 540 degrees/s")
    return args


def main():
    args = parse_args()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = args.output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    plans = {
        name: build_robot_plan(name, partition_x_min)
        for name, _, partition_x_min in ROBOT_SPECS
    }
    recorder = MetadataRecorder(
        run_directory / "metadata.json",
        plans,
        args,
    )

    stop_event = threading.Event()
    start_barrier = threading.Barrier(len(ROBOT_SPECS))
    workers = []
    motion_threads = []

    print("Output: {0}".format(run_directory))
    print(
        "Place robot_1 center at ({0:.4f}, {1:.4f}) m and robot_2 center "
        "at ({2:.4f}, {1:.4f}) m.".format(
            EDGE_MARGIN_METERS,
            EDGE_MARGIN_METERS,
            PARTITION_WIDTH_METERS + EDGE_MARGIN_METERS,
        )
    )
    print(
        "Point both robots along +x. They will rotate at each corner "
        "and move forward along their current heading."
    )
    print(
        "A return row intentionally reaches heading 180 deg through two "
        "separate 90 deg turns with a forward lane shift between them."
    )
    print("Press Ctrl+C to stop.")

    try:
        for name, serial_number, _ in ROBOT_SPECS:
            worker = RobotWorker(
                name=name,
                serial_number=serial_number,
                plan=plans[name],
                recorder=recorder,
                args=args,
                stop_event=stop_event,
                start_barrier=start_barrier,
            )
            worker.connect()
            workers.append(worker)

        if args.world_yaw is None:
            print(
                "Using each robot's startup yaw as its straight-ahead +x heading."
            )
            for worker in workers:
                worker.set_world_yaw(worker.initial_yaw)
        else:
            print(
                "Using --world-yaw ({0:.2f} deg) as the shared arena +x.".format(
                    args.world_yaw,
                )
            )
            for worker in workers:
                worker.set_world_yaw(args.world_yaw)
        recorder.record_positions(workers)

        for worker in workers:
            thread = threading.Thread(
                target=worker.run_plan,
                name="{0}-motion".format(worker.name),
            )
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

        complete = (
            not stop_event.is_set()
            and all(
                recorder.completed[name]
                for name, _, _ in ROBOT_SPECS
            )
        )
        reason = (
            "Both partition plans completed; the entire {0:.1f} m by "
            "{1:.1f} m area is covered.".format(
                AREA_X_METERS,
                AREA_Y_METERS,
            )
            if complete
            else "Coverage stopped before all partition plans completed."
        )
        if not complete:
            errors = [
                "{0}: {1}".format(name, recorder.errors[name])
                for name, _, _ in ROBOT_SPECS
                if recorder.errors[name]
            ]
            if errors:
                reason = "{0} {1}".format(reason, " | ".join(errors))
        recorder.finish(
            complete=complete,
            reason=reason,
            lane_count=len(make_lane_positions()),
        )
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
        recorder.finish(
            complete=False,
            reason="Interrupted by user.",
            lane_count=len(make_lane_positions()),
        )
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
        recorder.finish(
            complete=False,
            reason="Run failed: {0}".format(error),
            lane_count=len(make_lane_positions()),
        )
        print("Run failed: {0}".format(error))
    finally:
        stop_event.set()
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
