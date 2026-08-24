# -*-coding:utf-8-*-
"""Two-RoboMaster stigmergy search using camera-visible pheromone.

This keeps the robot setup, motion model, three-action camera-FOV action
space, boundary checks, telemetry, and image collection from random_walk.py.
Before every step, each robot divides its latest camera image into left,
straight, and right thirds and prefers the valid sector containing the least
pheromone-colored area. If every valid sector contains pheromone, the robot
falls back to a bounded random-walk action.
"""

import argparse
from datetime import datetime
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

import random_walk as base
from robomaster import media


# libmedia_codec's Windows H.264 decoder is not safe when both RoboMaster
# decoder threads enter it concurrently. Keep both streams active while
# serializing only the native decode call.
H264_DECODE_LOCK = threading.Lock()
ORIGINAL_H264_DECODE = media.LiveView._h264_decode


def serialized_h264_decode(liveview, data):
    with H264_DECODE_LOCK:
        return ORIGINAL_H264_DECODE(liveview, data)


media.LiveView._h264_decode = serialized_h264_decode


PARTITION_WIDTH_METERS = base.PARTITION_WIDTH_METERS
ROBOT_SPECS = (
    ("robot_1", "3JKCH8800100VW", 0.0),
    ("robot_2", "3JKCH8800100RC", PARTITION_WIDTH_METERS),
)


def build_partitioned_robot_start(name, partition_x_min):
    """Preserve this script's original separate half-arena setup."""
    return {
        "robot": name,
        "partition": {
            "x_min": partition_x_min,
            "x_max": partition_x_min + PARTITION_WIDTH_METERS,
            "y_min": 0.0,
            "y_max": base.AREA_Y_METERS,
        },
        "start_center": base.round_point(
            (
                partition_x_min + base.EDGE_MARGIN_METERS,
                base.EDGE_MARGIN_METERS,
            )
        ),
        "start_heading_deg": 0.0,
    }


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

SECTOR_NAMES = ("left", "straight", "right")
CAMERA_FRAME_WAIT_SECONDS = 2.0


class StigmergyRobotWorker(base.RobotWorker):
    """Robot worker that selects bounded actions from camera pheromone counts."""

    USE_ISOLATED_CAMERA_DECODER = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_ready = threading.Event()

    def capture_frames(self):
        """Read each stream once, sharing frames with control and image saving."""
        period = 1.0 / self.args.capture_fps
        next_capture = time.monotonic()

        while not self.stop_event.is_set():
            try:
                frame = self.ep_camera.read_cv2_image(
                    strategy="newest",
                    timeout=0.3,
                )
            except queue.Empty:
                continue
            except Exception as error:
                if not self.stop_event.is_set():
                    self.recorder.update_robot(
                        self.name,
                        error="Camera error: {0}".format(error),
                    )
                    self.stop_event.set()
                break

            if frame is None:
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()
            self.frame_ready.set()

            now = time.monotonic()
            if now < next_capture:
                continue
            next_capture = now + period

            self.frame_index += 1
            filename = "frame_{0:06d}_{1}.jpg".format(
                self.frame_index,
                datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")[:-3],
            )
            cv2.imwrite(str(self.image_directory / filename), frame)

    def pheromone_counts(self):
        """Return pheromone-pixel counts in left, straight, and right thirds."""
        if not self.frame_ready.wait(timeout=CAMERA_FRAME_WAIT_SECONDS):
            return None

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, PHEROMONE_HSV_LOWER, PHEROMONE_HSV_UPPER)
        width = mask.shape[1]
        boundaries = (0, width // 3, (2 * width) // 3, width)
        return {
            sector_name: int(
                cv2.countNonZero(mask[:, boundaries[index]:boundaries[index + 1]])
            )
            for index, sector_name in enumerate(SECTOR_NAMES)
        }

    def choose_stigmergy_action(self):
        """Choose the least-marked valid sector, or random walk if surrounded."""
        actions = self.valid_actions()
        counts = self.pheromone_counts()

        if not actions:
            # Match random_walk.py's corner recovery: rotate without translating
            # until at least one forward sector is inside the partition.
            action_name = self.random.choice(("left", "right"))
            turn_degrees = base.ACTION_TURNS[action_name]
            next_heading = base.normalize_angle_degrees(
                self.relative_heading + turn_degrees
            )
            return (
                action_name,
                turn_degrees,
                next_heading,
                0.0,
                counts,
                "boundary_recovery",
            )

        if counts is None:
            chosen = self.random.choice(actions)
            return (*chosen, None, "random_no_frame")

        valid_counts = [counts[action[0]] for action in actions]
        if all(count > 0 for count in valid_counts):
            chosen = self.random.choice(actions)
            return (*chosen, counts, "random_surrounded")

        minimum_count = min(valid_counts)
        least_marked = [
            action for action in actions if counts[action[0]] == minimum_count
        ]

        # As in robot_efficient.py, pheromone is the primary criterion and
        # lower turning cost breaks ties. Randomness remains only for exact ties.
        minimum_turn_cost = min(abs(action[1]) for action in least_marked)
        best_actions = [
            action
            for action in least_marked
            if abs(action[1]) == minimum_turn_cost
        ]
        chosen = self.random.choice(best_actions)
        return (*chosen, counts, "least_pheromone")

    def execute_stigmergy_step(self, step_index):
        (
            action_name,
            turn_degrees,
            next_heading,
            distance,
            counts,
            decision_mode,
        ) = self.choose_stigmergy_action()
        action = {
            "step": step_index,
            "name": action_name,
            "turn_deg": round(turn_degrees, 3),
            "distance_m": round(distance, 4),
            "decision_mode": decision_mode,
            "pheromone_pixels": counts,
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
                counts,
                turn_degrees,
                distance,
            )
        )

        self.command_stop()
        time.sleep(base.MOTION_SETTLE_SECONDS)
        self.rotate_to_yaw(self.target_sdk_yaw(next_heading))
        self.relative_heading = next_heading
        if distance > 0.0:
            self.drive_forward(
                distance,
                self.target_sdk_yaw(self.relative_heading),
            )

    def run_stigmergy_search(self):
        step_index = 0
        deadline = time.monotonic() + self.args.duration
        try:
            self.start_barrier.wait()
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self.execute_stigmergy_step(step_index)
                step_index += 1
            self.recorder.update_robot(self.name, completed=True)
        except Exception as error:
            print("{0} failed: {1}".format(self.name, error))
            self.recorder.update_robot(self.name, error=str(error))
            self.stop_event.set()
        finally:
            self.stop_motion()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run two bounded RoboMaster stigmergy-search robots that avoid "
            "camera-visible pheromone."
        )
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=base.DEFAULT_LINEAR_SPEED_MPS,
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=base.DEFAULT_TURN_SPEED_DPS,
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=base.DEFAULT_CAPTURE_FPS,
        help="JPEG capture rate per robot",
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
        name: build_partitioned_robot_start(name, partition_x_min)
        for name, _, partition_x_min in ROBOT_SPECS
    }
    recorder = base.MetadataRecorder(
        run_directory / "metadata.json",
        starts,
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
            starts["robot_1"]["start_center"]["x"],
            starts["robot_1"]["start_center"]["y"],
            starts["robot_2"]["start_center"]["x"],
        )
    )
    print(
        "Point both robots along +x. HSV pheromone range: {0} to {1}.".format(
            tuple(int(value) for value in PHEROMONE_HSV_LOWER),
            tuple(int(value) for value in PHEROMONE_HSV_UPPER),
        )
    )
    print("Press Ctrl+C to stop.")

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
            worker.connect()
            workers.append(worker)

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

        recorder.record_positions(workers)
        for worker in workers:
            thread = threading.Thread(
                target=worker.run_stigmergy_search,
                name="{0}-motion".format(worker.name),
            )
            motion_threads.append(thread)
            thread.start()

        stop_sent = False
        next_metadata_sample = (
            time.monotonic() + base.METADATA_SAMPLE_PERIOD_SECONDS
        )
        while any(thread.is_alive() for thread in motion_threads):
            for thread in motion_threads:
                thread.join(timeout=0.1)
            now = time.monotonic()
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
