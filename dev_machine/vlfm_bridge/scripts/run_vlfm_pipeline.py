"""Master script: orchestrates the full vlfm-native-mapping -> GO2_STACK
obstacle-avoidance loop on the real robot, with LIVE target-object
switching and a hard "lie down" stop.

Architecture (all pieces individually verified in earlier sessions):
  RGB-D (TCP, Jetson) --+
                         +--> ObstacleMap/ValueMap (vlfm native) --> best
  pose (UDP relay from   |    frontier --> /way_point (UDP relay) -->
  /state_estimation) ----+    localPlanner (obstacle-avoidance path) -->
                              pathFollower (autonomyMode=true) --> /cmd_vel
                              --> unitree_control --> WebRTC --> robot

Stop mechanism, revised after two live tests:
  1. PRIMARY (reliable, twice-confirmed live): kill pathFollower FIRST.
     This cuts off /cmd_vel at the source; both live tests so far show the
     robot stopping cleanly on its own once this happens, no "keeps
     executing the last command" behavior observed.
  2. SECONDARY / best-effort: only AFTER pathFollower is dead, call the
     /liedown service (std_srvs/Trigger -> SPORT_CMD["StandDown"]) to put
     the robot into an unambiguous resting state. NOT relied upon --
     live-tested and found unreliable twice: once "Data channel is not
     open" (stale WebRTC session), once a bare timeout with zero response
     even after 60s (likely DDS-service-call starvation under this
     project's typically high nav-stack CPU load -- laser_mapping_node
     alone measured at ~52% CPU, system load average ~17 -- possibly
     compounded by the zenoh-bridge-dds.service systemd unit's auto-restart
     loop churning the DDS discovery graph; that service targets a
     long-dead Tailscale link and should be `sudo systemctl stop`ped if
     not otherwise needed).
  Earlier idea (send a /way_point at the robot's own current position
  while still armed, hoping cmd_vel settles to zero on its own) was tried
  live and rejected: linear velocity did settle, but angular.z kept
  jittering (path-direction noise on a near-zero-distance goal) even
  though it didn't visibly move the robot that one time -- not trustworthy
  as a stop primitive.

INITIAL SCAN (2026-09-19): before pathFollower is even started, the script
rotates the robot a full 360deg closed-loop by commanding /cmd_vel yaw rate
directly through cmdvel_udp_relay.py (linear velocity forced to 0, |wz|
clamped, silent when the stream stops). Only after the scan is pathFollower
armed.

SAFETY:
  - Requires typed "ARM" confirmation before starting real motion.
  - Bounded runtime by default (--max-seconds), independent of Ctrl-C.
  - A human with the physical remote must be present for every real run --
    this script does not replace that.
  - Startup sends an initial /way_point at the robot's OWN current
    position before arming, so the first cycle can't lurch toward a stale
    default goal.

Usage:
    cd /home/tommy/Taowen/VLFM_Project/vlfm
    PYTHONPATH=$(pwd) /home/tommy/miniconda3/envs/vlfm/bin/python \\
        /home/tommy/Taowen/GO2_STACK/src/vlfm_bridge/scripts/run_vlfm_pipeline.py \\
        --target chair --max-seconds 300 [--record]

While running, type in this terminal (Enter to submit):
    target <name>     change the exploration target, e.g. "target sofa"
    stop               graceful stop (same as Ctrl-C): liedown, then exit
"""
import argparse
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import open_clip
import torch
import torch.nn.functional as _F

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import rho_theta
from vlfm.vlm.yolo_world import YOLOWorldClient, _caption_to_classes

# CGFM modules (dev_machine/cgfm/)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "cgfm"))
from scene_graph import LightweightSceneGraph
from semantic_map import compute_semantic_map
from frontier_scorer import select_frontier_by_score

JETSON_HOST = os.environ.get("GO2_JETSON_HOST", "192.168.3.18")
CAMERA_PORT = 6000
POSE_UDP_PORT = 8765
WAYPOINT_UDP_PORT = 8766
CMDVEL_UDP_PORT = 8767
WAYPOINT_MIN_REPUBLISH_DELTA_M = 0.05
LOOP_INTERVAL_S = 1.0
COMPOSITE_VIDEO_FPS = 1.5   # playback speed of the auto-generated composite video (ticks/s)

_CAMERA_FORWARD_OFFSET_M = 0.15
# Re-measured 2026-09-17 via my_tests-style floor-height-vs-distance
# regression against a live frame (see conversation/scratchpad
# calibrate_floor.py): at the OLD 8.6deg assumption, floor height computed
# 0.19m higher at 2m range than at 0.4m range (way off). pitch=0deg gave
# the flattest floor (-0.013m drift over 0.4-2.5m) -- the camera's actual
# mount changed (previously 8.6deg up / 0.385m off the ground) since the
# original go2_robot.py-era calibration; it now reads as level.
_CAMERA_PITCH_UP_DEG = 0.0

MIN_DEPTH = 0.3
MAX_DEPTH = 3.0
CAMERA_HEIGHT_M = 0.44  # re-measured 2026-09-17, see _CAMERA_PITCH_UP_DEG comment above
MIN_HEIGHT = -CAMERA_HEIGHT_M + 0.10
MAX_HEIGHT = 0.6
MAP_SIZE = 800
PIXELS_PER_METER = 20
FRONTIER_SEARCH_RADIUS = 1.0
DINO_CONF_THRESHOLD = 0.25
MAX_BOX_AREA_FRAC = 0.7
EDGE_CROP_FRAC = 0.08

RES_DIR = "/home/tommy/Taowen/RES"
# YOLO-World (vlfm/vlm/yolo_world.py) is open-vocabulary -- unlike YOLOv7 it
# isn't limited to the 80 COCO classes, so "fan"/"trash can" work again.
# Same ' . '-separated caption convention as the earlier GroundingDINO setup:
# a few synonym phrases per target helps recall.
DET_SYNONYMS = {
    "computer": "computer . computer tower . PC . monitor . gaming pc .",
    "fan": "fan . fans . ceiling fan . electric fan . cooling fan .",
    "chair": "chair . office chair . seat .",
    "person": "person . human .",
    "sofa": "sofa . couch .",
}

# PATCH (2026-09-20, wireless/router setup): every ROS 2 process spawned below (pathFollower, the UDP relays and
# the ros2 CLI calls such as /liedown) has to use the same DDS domain and CycloneDDS config as the Jetson and
# unitree_control, otherwise it silently lands in domain 0 and sees none of the wireless data. They only inherit
# the environment of the shell that started this script, so source the env script here instead of relying on
# the operator remembering to do it. GO2_ROS_ENV_SCRIPT=<path> selects another one (e.g. ros_env_mid360.sh for
# the old wired setup); GO2_ROS_ENV_SCRIPT="" adds nothing (original behaviour).
GO2_ROS_ENV_SCRIPT = os.environ.get(
    "GO2_ROS_ENV_SCRIPT", "/home/tommy/Taowen/autonomy_stack_go2/ros_env_wifi.sh"
)
ROS_ENV_SOURCE = (
    "source /opt/ros/jazzy/setup.bash && "
    "source /home/tommy/GO2_STACK_dev_ws/install/setup.bash"
    + (f" && source {GO2_ROS_ENV_SCRIPT} >/dev/null" if GO2_ROS_ENV_SCRIPT else "")
)
LOCAL_PLANNER_YAML = (
    "/home/tommy/GO2_STACK_dev_ws/install/local_planner/share/local_planner/"
    "config/unitree/unitree_go2_slow.yaml"
)
PATHFOLLOWER_MATCH = "install/local_planner/lib/local_planner/pathFollower"
POSE_RELAY_SCRIPT = "/home/tommy/Taowen/GO2_STACK/src/vlfm_bridge/scripts/pose_udp_relay.py"
WAYPOINT_RELAY_SCRIPT = "/home/tommy/Taowen/GO2_STACK/src/vlfm_bridge/scripts/waypoint_udp_relay.py"
CMDVEL_RELAY_SCRIPT = "/home/tommy/Taowen/GO2_STACK/src/vlfm_bridge/scripts/cmdvel_udp_relay.py"


def _ros_cli(cmd: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"{ROS_ENV_SOURCE} && {cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )


def _ros_popen(cmd: str, log_path: str = None) -> subprocess.Popen:
    # PATCH (2026-09-17): used to always silence output to DEVNULL, which
    # meant the relays' own per-message logging (useful for confirming
    # whether a given /way_point actually got sent) was unrecoverable after
    # the fact -- exactly the gap hit when diagnosing the 360-scan travelling
    # 2.9m in one direction instead of rotating. Callers that need that
    # visibility later should pass log_path.
    out = open(log_path, "a") if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        ["bash", "-c", f"{ROS_ENV_SOURCE} && exec {cmd}"],
        stdout=out, stderr=out,
    )


def _pkill(pattern: str) -> None:
    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def start_pathfollower(autonomy: bool, log_path: str = None) -> None:
    _pkill(PATHFOLLOWER_MATCH)
    time.sleep(1.0)
    mode = "true" if autonomy else "false"
    cmd = (
        f"ros2 run local_planner pathFollower --ros-args "
        f"--params-file {LOCAL_PLANNER_YAML} "
        f"-p useSerialPort:=false -p realRobot:=false -p autonomyMode:={mode} "
        # twoWayDrive:=false (2026-09-18): with it true, dirDiff>90deg made
        # pathFollower reverse instead of turning -- but the camera faces
        # forward, so driving backward means vlfm's own perception (which
        # is what actually decides where to explore next) never sees
        # anything new in the direction of travel, even though the lidar
        # can. Forcing forward-only makes localPlanner's path selection
        # rotate toward the goal (clamped to +/-95deg per tick, see
        # localPlanner.cpp's `if (!twoWayDrive)` joyDir clamp) instead of
        # backing up to it.
        f"-p sensorOffsetX:=0.2 -p sensorOffsetY:=0.0 -p twoWayDrive:=false "
        f"-p maxSpeed:=0.2 -p autonomySpeed:=0.2 -p maxYawRate:=30.0 "
        f"-p dirDiffThre:=1.0"
    )
    # PATCH (2026-09-17): this used to discard stdout to DEVNULL, silently
    # losing the [PF_DBG] instrumentation added to pathFollower.cpp while
    # root-causing the "won't turn until about to collide" bug -- pass
    # log_path so a real run captures it.
    _ros_popen(cmd, log_path=log_path)
    time.sleep(2.0)


def call_liedown() -> bool:
    """Hard stop: SPORT_CMD StandDown via the existing /liedown service.
    Overrides whatever Move state pathFollower/unitree_control were in."""
    try:
        result = _ros_cli('ros2 service call /liedown std_srvs/srv/Trigger "{}"', timeout=8.0)
        print(f"[liedown] {result.stdout.strip() or result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[liedown] service call timed out")
        return False


class WaypointUdpSender:
    FMT = "<3d"

    def __init__(self, port: int = WAYPOINT_UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)

    def send(self, x: float, y: float, z: float) -> None:
        self._sock.sendto(struct.pack(self.FMT, x, y, z), self._dest)


class CmdVelUdpSender:
    """Yaw-rate-only command stream to cmdvel_udp_relay.py (which forces linear
    velocity to 0, clamps |wz|, and goes silent when this stops sending)."""
    FMT = "<d"

    def __init__(self, port: int = CMDVEL_UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)

    def send(self, wz: float) -> None:
        self._sock.sendto(struct.pack(self.FMT, float(wz)), self._dest)


class PoseUdpClient:
    FMT = "<8d"
    SIZE = struct.calcsize(FMT)

    def __init__(self, port: int = POSE_UDP_PORT):
        self._lock = threading.Lock()
        self._xy_yaw = None
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            if len(data) != self.SIZE:
                continue
            _stamp, x, y, _z, qx, qy, qz, qw = struct.unpack(self.FMT, data)
            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            with self._lock:
                self._xy_yaw = (np.array([x, y]), yaw)

    def xy_yaw(self):
        with self._lock:
            return self._xy_yaw

    def close(self) -> None:
        self._stop.set()


class _RealSenseStreamClient:
    INTRINSICS_FMT = ">4fII f"
    INTRINSICS_SIZE = struct.calcsize(INTRINSICS_FMT)

    def __init__(self, host: str, port: int = 6000):
        self._host, self._port = host, port
        self._lock = threading.Lock()
        self._color = None
        self._depth = None
        self._intrinsics = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self._host, self._port))
                intr_bytes = self._recvall(sock, self.INTRINSICS_SIZE)
                if intr_bytes is not None:
                    fx, fy, ppx, ppy, width, height, depth_scale = struct.unpack(
                        self.INTRINSICS_FMT, intr_bytes
                    )
                    with self._lock:
                        self._intrinsics = {"fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy,
                                             "width": width, "height": height, "depth_scale": depth_scale}
                while not self._stop.is_set():
                    header = self._recvall(sock, 8)
                    if header is None:
                        break
                    color_len, depth_len = struct.unpack(">II", header)
                    color_bytes = self._recvall(sock, color_len)
                    depth_bytes = self._recvall(sock, depth_len)
                    if color_bytes is None or depth_bytes is None:
                        break
                    color = cv2.imdecode(np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    depth = cv2.imdecode(np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    with self._lock:
                        self._color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        self._depth = depth
                sock.close()
            except (socket.timeout, ConnectionRefusedError, OSError):
                pass
            if not self._stop.is_set():
                time.sleep(1.0)

    def intrinsics(self):
        with self._lock:
            return self._intrinsics

    def latest(self):
        with self._lock:
            return self._color, self._depth

    def close(self) -> None:
        self._stop.set()


def _build_camera_to_body_transform() -> np.ndarray:
    theta = np.deg2rad(_CAMERA_PITCH_UP_DEG)
    c, s = np.cos(theta), np.sin(theta)
    t = np.eye(4)
    t[0, 0], t[0, 2] = c, -s
    t[2, 0], t[2, 2] = s, c
    t[0, 3] = _CAMERA_FORWARD_OFFSET_M
    return t


_CAMERA_TO_BODY = _build_camera_to_body_transform()


def get_camera_transform(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    body_to_episodic = np.eye(4)
    body_to_episodic[0, 0], body_to_episodic[0, 1] = c, -s
    body_to_episodic[1, 0], body_to_episodic[1, 1] = s, c
    body_to_episodic[0, 3], body_to_episodic[1, 3] = xy[0], xy[1]
    return body_to_episodic @ _CAMERA_TO_BODY


def normalize_depth(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale
    normalized = np.clip((depth_m - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0.0, 1.0)
    normalized[depth_raw == 0] = 0.0
    edge_px = int(depth_raw.shape[1] * EDGE_CROP_FRAC)
    if edge_px > 0:
        normalized[:, :edge_px] = 0.0
        normalized[:, -edge_px:] = 0.0
    return normalized


def box_area_frac(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


TARGET_LOCK_WINDOW = 5         # sliding window of the last N perception ticks...
TARGET_LOCK_MIN_HITS = 3       # ...lock the target once at least this many of them saw it
TARGET_STOP_RADIUS_M = 0.25    # stop closing in once this near the locked target position
TARGET_DEPTH_PATCH_PX = 4      # half-width of the median-depth sampling patch around the box center
TARGET_MAX_DIST_M = 2.5        # a detection whose box-center depth is beyond this is 'Found but too far': not counted (far depth-based xy is too inaccurate)


def _tick_walk(idx, camera, pose_client, goal_xy, run_dir, video_writer):
    """Post-lock tick: NO detector / BLIP2 / map updates / frontier logic --
    just read the pose for the distance-to-goal check and keep recording the
    camera view (frame + depth + video) so the run stays reviewable. Returns
    (xy, dist) or None if pose/camera weren't ready."""
    xy_yaw = pose_client.xy_yaw()
    color, depth_raw = camera.latest()
    if xy_yaw is None or color is None or depth_raw is None:
        return None
    xy = xy_yaw[0]
    dist = float(np.hypot(goal_xy[0] - xy[0], goal_xy[1] - xy[1]))
    vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"), depth_raw)
    cv2.putText(vis, f"WALKING TO GOAL ({goal_xy[0]:.2f},{goal_xy[1]:.2f}) dist={dist:.2f}m",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(run_dir, "frames", f"tick_{idx:05d}.jpg"), vis)
    if video_writer is not None:
        video_writer.write(vis)
    return xy, dist


def draw_detections(color_bgr, target, score, detections, valid_idxs):
    """Draws every YOLO-World box onto a copy of the frame (green if it
    passed the confidence+area filters, gray otherwise) plus the BLIP2ITM
    score, so a run can be visually audited frame-by-frame to see what the
    model actually detected."""
    vis = color_bgr.copy()
    height, width = vis.shape[:2]
    if detections is not None:
        for i, (box, logit, phrase) in enumerate(
            zip(detections.boxes, detections.logits, detections.phrases)
        ):
            x1, y1, x2, y2 = box.numpy() if hasattr(box, "numpy") else box
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            ok = i in valid_idxs
            color = (0, 220, 0) if ok else (120, 120, 120)
            cv2.rectangle(vis, pt1, pt2, color, 2 if ok else 1)
            label = f"{phrase} {float(logit):.2f}"
            cv2.putText(vis, label, (pt1[0], max(0, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    header = f"target={target!r} blip_score={score:+.3f}"
    cv2.putText(vis, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis


def box_center_depth_m(box, depth_raw, depth_scale, width, height):
    """Median depth (m) over a small patch around the box center, or None if
    the patch has no valid depth."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2 * width)
    cy = int((y1 + y2) / 2 * height)
    y0, y1p = max(0, cy - TARGET_DEPTH_PATCH_PX), min(height, cy + TARGET_DEPTH_PATCH_PX + 1)
    x0, x1p = max(0, cx - TARGET_DEPTH_PATCH_PX), min(width, cx + TARGET_DEPTH_PATCH_PX + 1)
    patch = depth_raw[y0:y1p, x0:x1p]
    valid = patch[patch != 0]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid)) * depth_scale
    return depth_m if depth_m > 0 else None


def filter_by_distance(detections, valid_idxs, depth_raw, depth_scale, width, height, idx, log):
    """Distance gate on detections that already passed the class/confidence/
    area filters: keep only those whose box-center depth is <= TARGET_MAX_DIST_M.
    A far detection ("Found but too far") or one with no valid depth (distance
    can't be verified) is logged and NOT counted, so it can neither add a lock
    hit nor produce a far, inaccurate goal estimate."""
    kept = []
    for i in valid_idxs:
        box = detections.boxes[i]
        box = box.numpy() if hasattr(box, "numpy") else box
        d = box_center_depth_m(box, depth_raw, depth_scale, width, height)
        what = f"{detections.phrases[i]!r} conf={float(detections.logits[i]):.2f}"
        if d is None:
            log(f"  [tick {idx}] Found but no valid depth (distance can't be verified): {what} -- not counted")
        elif d > TARGET_MAX_DIST_M:
            log(f"  [tick {idx}] Found but too far: {what} dist={d:.2f}m > {TARGET_MAX_DIST_M:.1f}m -- not counted")
        else:
            kept.append(i)
    return kept


def estimate_target_xy(box, depth_raw, depth_scale, fx, fy, width, height, tf_camera_to_episodic):
    """Rough world-frame (x, y) of a detected target: box-center pixel's
    median depth (over a small patch, for robustness against single-pixel
    noise/no-return) back-projected through the pinhole model and the
    camera->episodic transform -- deliberately simple (no SAM segmentation
    or point-cloud clustering like vlfm_navigator_node.py's ObjectPointCloudMap),
    per the "just use the underlying depth nav" request. Returns None if the
    patch has no valid depth."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2 * width)
    cy = int((y1 + y2) / 2 * height)
    depth_m = box_center_depth_m(box, depth_raw, depth_scale, width, height)
    if depth_m is None:
        return None
    x_cam = (cx - width / 2) * depth_m / fx
    y_cam = (cy - height / 2) * depth_m / fy
    point_cam = np.array([depth_m, -x_cam, -y_cam, 1.0])  # (forward, left, up), matches get_point_cloud's convention
    point_world = tf_camera_to_episodic @ point_cam
    return point_world[:2]


class StdinCommands(threading.Thread):
    """Background reader for 'target <name>' / 'stop' typed into this
    terminal while the main loop runs, so the target object can change
    without restarting the whole pipeline."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q: "queue.Queue[str]" = queue.Queue()

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if line:
                self.q.put(line)

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


SCAN_STEPS = 8                 # 8 x 45deg = one full turn, then back at the start heading
SCAN_ROT_KP = 1.2              # rad/s of yaw rate per rad of yaw error
SCAN_ROT_MAX_WZ = 0.5          # rad/s (~29deg/s) -- relay clamps at 0.6 regardless
SCAN_ROT_MIN_WZ = 0.25         # rad/s, stay above the Go2's small-command deadband while |err| > tol
SCAN_ROT_TOL_DEG = 8.0         # step counts as reached within this, held for 3 loop ticks
SCAN_ROT_LOOP_HZ = 20.0
SCAN_ROT_STEP_TIMEOUT_S = 10.0
SCAN_ROT_RESPONSE_CHECK_S = 2.5    # by now the yaw error must have shrunk by >= MIN_PROGRESS...
SCAN_ROT_MIN_PROGRESS_DEG = 5.0    # ...or the rotation isn't working (wrong sign / robot not
                                   # responding / frozen pose): stop and abort the scan
SCAN_PERCEIVE_TICKS = 2        # perception ticks captured at rest after each step
SCAN_MAX_DRIFT_M = 1.0         # rotation-only must not translate; abort if it does


def _tick_perception(idx, target, camera, pose_client, obstacle_map, scene_graph,
                      detector, intr, fx, fy, fov, run_dir, log,
                      video_writer=None):
    """One perception+mapping tick, shared by the initial 360-scan and the
    main explore loop: capture a frame, save it, run YOLO-World detection,
    update the obstacle map + scene graph, save snapshots. Returns None if
    pose/camera data wasn't ready yet this tick."""
    det_caption = DET_SYNONYMS.get(target, f"{target} .")

    xy_yaw = pose_client.xy_yaw()
    color, depth_raw = camera.latest()
    if xy_yaw is None or color is None or depth_raw is None:
        return None
    xy, yaw = xy_yaw
    depth_norm = normalize_depth(depth_raw, intr["depth_scale"])
    tf_camera_to_episodic = get_camera_transform(xy, yaw)

    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"), depth_raw)
    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLORMAP_JET,
    )
    depth_vis[depth_raw == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}_vis.jpg"), depth_vis)

    # Defense-in-depth against a real failure mode hit live (2026-09-18): a
    # long-running YOLO-World server that's had set_classes() called with
    # many different class lists over a session can end up returning boxes
    # labeled with some OTHER class entirely (e.g. "person"/"tv"/"microwave")
    # even though it was just asked for only `det_classes` -- confirmed via a
    # fresh model instance returning zero detections on the same frame where
    # the long-lived server confidently returned unrelated classes. Restart
    # the server if this recurs; this check just stops a mislabeled box from
    # being silently treated as "found the target" regardless of the cause.
    det_classes = set(_caption_to_classes(det_caption))
    try:
        detections = detector.predict(color, caption=det_caption)
        valid_idxs = [
            i for i, (box, logit, phrase) in enumerate(
                zip(detections.boxes, detections.logits, detections.phrases)
            )
            if phrase in det_classes
            and logit >= DINO_CONF_THRESHOLD and box_area_frac(box) <= MAX_BOX_AREA_FRAC
        ]
    except Exception as e:
        log(f"  [warn] YOLO-World failed: {e}")
        detections, valid_idxs = None, []
    valid_idxs = filter_by_distance(detections, valid_idxs, depth_raw, intr["depth_scale"],
                                    intr["width"], intr["height"], idx, log)
    target_seen = len(valid_idxs) > 0

    score = 0.0  # BLIP2ITM removed; kept for draw_detections header only
    if valid_idxs and scene_graph is not None:
        try:
            scene_graph.update(
                color, detections, valid_idxs,
                depth_raw, intr["depth_scale"],
                tf_camera_to_episodic, fx, fy,
            )
        except Exception as e:
            log(f"  [warn] scene_graph.update failed: {e}")

    det_vis = draw_detections(color_bgr, target, score, detections, valid_idxs)
    cv2.imwrite(os.path.join(run_dir, "frames", f"tick_{idx:05d}.jpg"), det_vis)
    if video_writer is not None:
        video_writer.write(det_vis)

    try:
        obstacle_map.update_map(
            depth=depth_norm, tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, fx=fx, fy=fy,
            topdown_fov=fov, explore=True, update_obstacles=True,
        )
        obstacle_map.update_agent_traj(xy, yaw)
    except IndexError as e:
        log(f"  [warn] pose outside map bounds: {e}")

    cv2.imwrite(os.path.join(run_dir, "occupancy_map", f"tick_{idx:05d}.png"), obstacle_map.visualize())

    return {
        "xy": xy, "yaw": yaw, "score": score, "target_seen": target_seen,
        "valid_idxs": valid_idxs, "detections": detections,
        "frontiers": obstacle_map.frontiers,
        "depth_raw": depth_raw, "tf_camera_to_episodic": tf_camera_to_episodic,
    }


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def rotate_to_yaw(target_yaw, pose_client, cmdvel_sender, xy0, should_stop):
    """Closed-loop in-place rotation to an absolute SLAM yaw, by commanding the
    yaw rate directly (see cmdvel_udp_relay.py for why the old waypoint-based
    turning was replaced). Yaw error is the median of the last 5 samples so a
    single SLAM glitch can't decide anything. Always leaves a zero command
    behind. Returns (outcome, final_err_rad) with outcome in
    ok | timeout | no_response | drift | stopped | no_pose."""
    dt = 1.0 / SCAN_ROT_LOOP_HZ
    tol = np.deg2rad(SCAN_ROT_TOL_DEG)
    progress_min = np.deg2rad(SCAN_ROT_MIN_PROGRESS_DEG)
    errs = deque(maxlen=5)
    t0 = time.time()
    err0, ok_n, drift_n, err = None, 0, 0, 0.0
    try:
        while True:
            if should_stop():
                return "stopped", err
            py = pose_client.xy_yaw()
            if py is None:
                if time.time() - t0 > 3.0:
                    return "no_pose", err
                time.sleep(dt)
                continue
            xy, yaw = py
            errs.append(_wrap_pi(target_yaw - yaw))
            err = float(np.median(errs))
            if err0 is None:
                err0 = abs(err)
            elapsed = time.time() - t0

            drift_n = drift_n + 1 if float(np.hypot(xy[0] - xy0[0], xy[1] - xy0[1])) > SCAN_MAX_DRIFT_M else 0
            if drift_n >= int(SCAN_ROT_LOOP_HZ * 0.5):
                return "drift", err
            ok_n = ok_n + 1 if abs(err) < tol else 0
            if ok_n >= 3:
                return "ok", err
            if (elapsed >= SCAN_ROT_RESPONSE_CHECK_S and err0 > tol + progress_min
                    and (err0 - abs(err)) < progress_min):
                return "no_response", err
            if elapsed > SCAN_ROT_STEP_TIMEOUT_S:
                return "timeout", err

            wz = float(np.clip(SCAN_ROT_KP * err, -SCAN_ROT_MAX_WZ, SCAN_ROT_MAX_WZ))
            if abs(wz) < SCAN_ROT_MIN_WZ:
                wz = float(np.sign(err)) * SCAN_ROT_MIN_WZ
            cmdvel_sender.send(wz)
            time.sleep(dt)
    finally:
        cmdvel_sender.send(0.0)


def _stop_rotation(cmdvel_sender) -> None:
    """Explicit zero-rate burst, then let the robot come to rest before the
    next camera capture (also lets cmdvel_udp_relay finish its own zero burst
    and go silent -- it must be quiet before pathFollower takes /cmd_vel)."""
    for _ in range(6):
        cmdvel_sender.send(0.0)
        time.sleep(0.05)
    time.sleep(0.6)


def perform_initial_scan(xy0, yaw0, camera, pose_client, cmdvel_sender,
                          obstacle_map, scene_graph, detector,
                          intr, fx, fy, fov, run_dir, log, target, frame_idx,
                          video_writer=None, should_stop=lambda: False):
    """Turn a full 360 degrees in SCAN_STEPS steps before exploring, so the
    obstacle/value maps already cover every direction from the start spot.

    Runs BEFORE pathFollower is started: this function alone owns /cmd_vel
    (via cmdvel_udp_relay.py, yaw rate only), rotating closed-loop to absolute
    headings yaw0 + k*360/SCAN_STEPS and capturing/mapping SCAN_PERCEIVE_TICKS
    frames at rest after each step (sharper than capturing mid-turn). The last
    step brings the robot back to its original heading.

    Aborts (stops rotating, returns early) if the rotation doesn't respond,
    the robot drifts > SCAN_MAX_DRIFT_M, or two steps in a row time out.
    Returns the updated frame_idx."""
    log(f"Initial 360-degree scan: {SCAN_STEPS} steps of {360 / SCAN_STEPS:.0f}deg, closed-loop yaw control "
        f"(pathFollower not running; tol {SCAN_ROT_TOL_DEG:.0f}deg, max {np.rad2deg(SCAN_ROT_MAX_WZ):.0f}deg/s, "
        f"abort if drift from start exceeds {SCAN_MAX_DRIFT_M}m)...")

    def perceive(step_label):
        nonlocal frame_idx
        for _ in range(SCAN_PERCEIVE_TICKS):
            tick_t0 = time.time()
            frame_idx += 1
            result = _tick_perception(frame_idx, target, camera, pose_client, obstacle_map,
                                       scene_graph, detector, intr, fx, fy, fov,
                                       run_dir, log, video_writer)
            if result is None:
                frame_idx -= 1
            else:
                drift = float(np.hypot(result["xy"][0] - xy0[0], result["xy"][1] - xy0[1]))
                log(f"  [scan {step_label} tick {frame_idx}] pose=({result['xy'][0]:+.2f},{result['xy'][1]:+.2f}) "
                    f"yaw={np.rad2deg(result['yaw']):+.0f}deg frontiers={len(result['frontiers'])} "
                    f"drift_from_start={drift:.2f}m")
            elapsed = time.time() - tick_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)

    perceive(f"0/{SCAN_STEPS} (start heading {np.rad2deg(yaw0):+.0f}deg)")
    timeouts_in_a_row = 0
    for step in range(1, SCAN_STEPS + 1):
        target_yaw = _wrap_pi(yaw0 + step * (2 * np.pi / SCAN_STEPS))
        t_step = time.time()
        outcome, err = rotate_to_yaw(target_yaw, pose_client, cmdvel_sender, xy0, should_stop)
        _stop_rotation(cmdvel_sender)
        log(f"  [scan {step}/{SCAN_STEPS}] rotate to {np.rad2deg(target_yaw):+.0f}deg: {outcome} "
            f"(final yaw error {np.rad2deg(err):+.1f}deg, {time.time() - t_step:.1f}s)")
        if outcome in ("no_response", "drift", "stopped", "no_pose"):
            log(f"  [SCAN ABORTED] rotation outcome {outcome!r} -- robot left stationary.")
            return frame_idx
        timeouts_in_a_row = timeouts_in_a_row + 1 if outcome == "timeout" else 0
        if timeouts_in_a_row >= 2:
            log("  [SCAN ABORTED] two rotation steps in a row timed out -- robot left stationary.")
            return frame_idx
        if step < SCAN_STEPS:
            perceive(f"{step}/{SCAN_STEPS}")
    log("Scan complete (back at the start heading).")
    return frame_idx


def next_run_dir(base_dir: str) -> str:
    path = os.path.join(base_dir, "vlfm_pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    for sub in ("frames", "occupancy_map", "depth"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="chair")
    parser.add_argument("--max-seconds", type=float, default=300.0,
                         help="Hard runtime cap regardless of Ctrl-C (safety net).")
    parser.add_argument("--skip-confirm", action="store_true",
                         help="Skip the typed ARM confirmation (do not use unattended).")
    parser.add_argument("--record", action="store_true",
                         help="Record the first-person camera feed (with detection "
                              "overlay) to first_person.mp4 in the run directory.")
    args = parser.parse_args()

    run_dir = next_run_dir(RES_DIR)
    log_path = os.path.join(run_dir, "log.txt")
    log_f = open(log_path, "a")

    def log(msg: str) -> None:
        # Colored in the terminal only (not in the file -- ANSI codes would
        # just be garbage bytes in log.txt): yellow for a tick that actually
        # saw the target, green for the final FOUND/stop line, so both stand
        # out against the frontier-exploration noise while watching a run
        # live.
        if msg.startswith("FOUND "):
            print(f"\033[32m{msg}\033[0m")
        elif ">>> saw " in msg:
            print(f"\033[33m{msg}\033[0m")
        else:
            print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    print("=" * 70)
    print("VLFM full-pipeline master script -- REAL ROBOT MOTION")
    print("Stop mechanism: kill pathFollower (primary), /liedown best-effort (secondary).")
    print("A human MUST be holding the physical remote right now.")
    print("=" * 70)
    if not args.skip_confirm:
        confirmation = input("Type ARM to proceed, anything else aborts: ").strip()
        if confirmation != "ARM":
            print("Aborted, nothing started.")
            return

    state = {"target": args.target, "stop": False}
    state_lock = threading.Lock()

    log("Cleaning up any stale relay/pathFollower processes...")
    _pkill("pose_udp_relay.py")
    _pkill("waypoint_udp_relay.py")
    _pkill("cmdvel_udp_relay.py")
    _pkill(PATHFOLLOWER_MATCH)
    time.sleep(1.0)

    log("Starting pose_udp_relay.py + waypoint_udp_relay.py + cmdvel_udp_relay.py (rotation-only) ...")
    pose_relay_proc = _ros_popen(
        f"/usr/bin/python3 {POSE_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "pose_udp_relay.log"))
    waypoint_relay_proc = _ros_popen(
        f"/usr/bin/python3 {WAYPOINT_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "waypoint_udp_relay.log"))
    cmdvel_relay_proc = _ros_popen(
        f"/usr/bin/python3 {CMDVEL_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "cmdvel_udp_relay.log"))
    time.sleep(2.0)

    camera = _RealSenseStreamClient(JETSON_HOST, CAMERA_PORT)
    pose_client = PoseUdpClient(POSE_UDP_PORT)
    waypoint_sender = WaypointUdpSender(WAYPOINT_UDP_PORT)
    cmdvel_sender = CmdVelUdpSender(CMDVEL_UDP_PORT)

    log("Waiting for camera intrinsics + first pose sample...")
    t0 = time.time()
    while camera.intrinsics() is None or pose_client.xy_yaw() is None:
        if time.time() - t0 > 30.0:
            log("ABORT: no camera or pose data within 30s. Not arming.")
            _pkill(PATHFOLLOWER_MATCH)
            return
        time.sleep(0.2)
    intr = camera.intrinsics()
    fx, fy, width = intr["fx"], intr["fy"], intr["width"]
    fov = 2 * np.arctan(width / (2 * fx))
    log(f"Camera OK (fx={fx:.1f}, fov={np.rad2deg(fov):.1f}deg). Pose OK.")

    video_writer = None
    if args.record:
        video_path = os.path.join(run_dir, "first_person.mp4")
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / LOOP_INTERVAL_S, (intr["width"], intr["height"]),
        )
        log(f"Recording first-person view to {video_path}")

    xy0, yaw0 = pose_client.xy_yaw()
    log(f"Pinning startup goal to current position {tuple(xy0.round(3))} before arming...")
    for _ in range(3):
        waypoint_sender.send(float(xy0[0]), float(xy0[1]), 0.0)
        time.sleep(0.2)

    obstacle_map = ObstacleMap(
        min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, agent_radius=0.2,
        area_thresh=0.5, hole_area_thresh=-1, size=MAP_SIZE, pixels_per_meter=PIXELS_PER_METER,
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    clip_model = clip_model.cuda().eval()
    scene_graph = LightweightSceneGraph(clip_model, clip_preprocess, device="cuda")
    cached_target: str = ""
    cached_text_feat = None
    detector = YOLOWorldClient(port=12185)

    stdin_reader = StdinCommands()
    stdin_reader.start()

    def request_stop(signum=None, frame=None) -> None:
        _pkill(PATHFOLLOWER_MATCH)  # cut /cmd_vel NOW, even if the main thread is stuck in a blocking VLM call
        with state_lock:
            state["stop"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Everything from here on (scan included) must go through the same
    # kill-pathFollower-then-liedown cleanup in `finally` -- start_pathfollower
    # is called INSIDE the try so a Ctrl-C mid-scan still stops the robot
    # via the normal path instead of an uncaught KeyboardInterrupt leaving
    # it armed.
    frame_idx = 0
    last_way_point = None
    hit_window = deque(maxlen=TARGET_LOCK_WINDOW)  # (seen, world-xy estimate or None) per tick
    target_lock_xy = None
    def should_stop() -> bool:
        with state_lock:
            return state["stop"]

    try:
        # The scan runs FIRST, with pathFollower deliberately NOT started: the
        # scan alone owns /cmd_vel (yaw rate only, via cmdvel_udp_relay.py).
        frame_idx = perform_initial_scan(
            xy0, yaw0, camera, pose_client, cmdvel_sender, obstacle_map, scene_graph,
            detector, intr, fx, fy, fov, run_dir, log, state["target"], 0,
            video_writer, should_stop,
        )
        if should_stop():
            log("Stop requested during the scan -- not arming pathFollower.")
            return

        # Hand /cmd_vel over to pathFollower: pin its goal to wherever the robot
        # is NOW (so it can't lurch toward a stale goal), give the relay time to
        # finish its zero burst and go silent, then arm.
        cur_xy, _cur_yaw = pose_client.xy_yaw()
        for _ in range(3):
            waypoint_sender.send(float(cur_xy[0]), float(cur_xy[1]), 0.0)
            time.sleep(0.2)
        time.sleep(0.6)
        log("Starting pathFollower with autonomyMode=true (ARMED)...")
        start_pathfollower(autonomy=True, log_path=os.path.join(run_dir, "pathFollower_debug.log"))

        start_time = time.time()
        log(f"ARMED. target={state['target']!r}. Type 'target <name>' or 'stop' anytime. "
            f"Hard cap {args.max_seconds:.0f}s (scan time not counted against this).")

        while True:
            with state_lock:
                if state["stop"]:
                    break
            if time.time() - start_time > args.max_seconds:
                log("Max runtime reached, stopping.")
                break

            cmd = stdin_reader.poll()
            if cmd:
                if cmd == "stop":
                    with state_lock:
                        state["stop"] = True
                    break
                elif cmd.startswith("target "):
                    new_target = cmd[len("target "):].strip()
                    if new_target:
                        with state_lock:
                            state["target"] = new_target
                        hit_window.clear()
                        target_lock_xy = None
                        last_way_point = None
                        cached_target = ""   # force text-feat re-encode on next tick
                        # scene_graph intentionally NOT cleared: history persists
                        log(f">>> target changed to {new_target!r}")
                else:
                    log(f"(unrecognized command: {cmd!r})")

            loop_t0 = time.time()
            frame_idx += 1
            target = state["target"]

            if target_lock_xy is not None:
                # LOCKED: the upper layer does no more reasoning at all -- no
                # detection, no BLIP2, no map update, no frontier choice, no new
                # waypoints. localPlanner/pathFollower own getting to the goal;
                # this only reads the pose to report/check the distance.
                walk = _tick_walk(frame_idx, camera, pose_client, target_lock_xy, run_dir, video_writer)
                if walk is None:
                    frame_idx -= 1
                    time.sleep(LOOP_INTERVAL_S)
                    continue
                xy, dist_to_target = walk
                log(f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s target={target!r}] "
                    f"pose=({xy[0]:+.2f},{xy[1]:+.2f})  Walking toward the goal "
                    f"({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) dist={dist_to_target:.2f}m")
                if dist_to_target <= TARGET_STOP_RADIUS_M:
                    log(f"FOUND {target!r} at ({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}), "
                        f"{dist_to_target:.2f}m away -- stopping early.")
                    with state_lock:
                        state["stop"] = True
                    break
                elapsed = time.time() - loop_t0
                if elapsed < LOOP_INTERVAL_S:
                    time.sleep(LOOP_INTERVAL_S - elapsed)
                continue

            result = _tick_perception(frame_idx, target, camera, pose_client, obstacle_map,
                                       scene_graph, detector, intr, fx, fy, fov,
                                       run_dir, log, video_writer)
            if result is None:
                frame_idx -= 1
                time.sleep(LOOP_INTERVAL_S)
                continue
            xy, yaw = result["xy"], result["yaw"]
            frontiers = result["frontiers"]
            status = (f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s target={target!r}] "
                      f"pose=({xy[0]:+.2f},{xy[1]:+.2f}) score={result['score']:+.3f} "
                      f"frontiers={len(frontiers)}")

            fresh_xy = None
            if result["target_seen"]:
                valid_idxs, detections = result["valid_idxs"], result["detections"]
                best_idx = max(valid_idxs, key=lambda i: detections.logits[i])
                box = detections.boxes[best_idx].numpy()
                fresh_xy = estimate_target_xy(
                    box, result["depth_raw"], intr["depth_scale"], fx, fy,
                    intr["width"], intr["height"], result["tf_camera_to_episodic"],
                )
            hit_window.append((bool(result["target_seen"]), fresh_xy))
            hits = sum(1 for seen, _ in hit_window if seen)
            if result["target_seen"]:
                status += (f"  >>> saw {target!r} conf={detections.logits[best_idx]:.3f} "
                           f"hits={hits}/{len(hit_window)} (lock at {TARGET_LOCK_MIN_HITS} "
                           f"within the last {TARGET_LOCK_WINDOW} ticks)")

            # Lock rule: >= TARGET_LOCK_MIN_HITS detections among the last
            # TARGET_LOCK_WINDOW ticks. The goal is the MEDIAN of those ticks'
            # depth-based world-frame estimates (robust to one bad depth patch),
            # sent once as a fixed waypoint; from then on the branch above takes
            # over and the upper layer stays out of the way.
            if hits >= TARGET_LOCK_MIN_HITS:
                estimates = [p for seen, p in hit_window if seen and p is not None]
                if estimates:
                    target_lock_xy = np.median(np.array(estimates), axis=0)
                    waypoint_sender.send(float(target_lock_xy[0]), float(target_lock_xy[1]), 0.0)
                    last_way_point = (float(target_lock_xy[0]), float(target_lock_xy[1]))
                    dist0 = float(np.hypot(target_lock_xy[0] - xy[0], target_lock_xy[1] - xy[1]))
                    per_frame = " ".join(f"({e[0]:.2f},{e[1]:.2f})" for e in estimates)
                    status += (f"  LOCKED goal=({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) "
                               f"dist={dist0:.2f}m [median of {len(estimates)} estimates: {per_frame}] "
                               f"-> handed to nav stack, no more upper-layer decisions")
                else:
                    status += "  [warn] enough hits but no valid depth estimate yet"

            if target_lock_xy is None and len(frontiers) > 0:
                # Re-encode text feat only when target changes
                if target != cached_target:
                    cached_text_feat = scene_graph.encode_text(target)
                    cached_target = target
                scored_objects = scene_graph.get_scored_objects(cached_text_feat)
                sem_map = compute_semantic_map(obstacle_map, scored_objects)
                best_idx = select_frontier_by_score(obstacle_map, xy, sem_map)
                best = frontiers[best_idx]
                rho, theta = rho_theta(xy, yaw, best)
                status += f"  best_frontier=({best[0]:.2f},{best[1]:.2f}) turn={np.rad2deg(theta):+.0f}deg dist={rho:.2f}m"
                moved = last_way_point is None or np.hypot(
                    best[0]-last_way_point[0], best[1]-last_way_point[1]
                ) >= WAYPOINT_MIN_REPUBLISH_DELTA_M
                if moved:
                    waypoint_sender.send(float(best[0]), float(best[1]), 0.0)
                    last_way_point = (float(best[0]), float(best[1]))
                    status += "  [/way_point sent]"
            log(status)

            elapsed = time.time() - loop_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)
    finally:
        for _ in range(3):  # if we die mid-rotation, leave an explicit zero yaw rate behind
            cmdvel_sender.send(0.0)
            time.sleep(0.03)
        log("Stopping: killing pathFollower first (primary, reliable stop)...")
        _pkill(PATHFOLLOWER_MATCH)
        _pkill("cmdvel_udp_relay.py")
        time.sleep(1.0)  # let the /cmd_vel stream actually cease
        log("Best-effort /liedown (secondary -- NOT guaranteed under load, see module docstring)...")
        call_liedown()
        camera.close()
        pose_client.close()
        if video_writer is not None:
            video_writer.release()
            log(f"First-person recording saved to {os.path.join(run_dir, 'first_person.mp4')}")
        cv2.imwrite(os.path.join(run_dir, "go2_obstacle_map_final.png"), obstacle_map.visualize())
        if video_writer is not None:
            # Robot is already stopped/lying down by now, so this (a few seconds
            # of encoding) can't delay anything safety-relevant. Turns the raw
            # first-person recording into the composite (camera + occupancy map
            # + value map, time-aligned); the raw one is kept as
            # first_person_orig.mp4. Never allowed to break cleanup.
            log("Building composite review video (camera + occupancy map)...")
            try:
                out = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "make_composite_video.py"),
                     run_dir, "--fps", str(COMPOSITE_VIDEO_FPS)],
                    capture_output=True, text=True, timeout=300)
                log((out.stdout.strip() or out.stderr.strip() or "(no output)")[-400:])
            except Exception as e:  # noqa: BLE001
                log(f"[warn] composite video failed ({e}); raw recording is still in first_person.mp4")
        log(f"Done. {frame_idx} ticks. Results in {run_dir}")
        log("pathFollower is dead -> /cmd_vel has stopped (primary stop, confirmed reliable). "
            "/liedown above may or may not have actually landed the robot -- check the "
            "[liedown] line and the robot itself; don't assume it's lying down.")
        log_f.close()


if __name__ == "__main__":
    main()
