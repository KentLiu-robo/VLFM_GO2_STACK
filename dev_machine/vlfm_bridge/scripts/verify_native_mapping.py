"""Verifies vlfm's OWN (i.e. "native", not the lidar-BEV substitute used by
vlfm_bridge/vlfm_navigator_node.py) camera-depth mapping pipeline --
vlfm.mapping.obstacle_map.ObstacleMap + vlfm.mapping.value_map.ValueMap --
running against GO2_STACK as the lower nav stack on the real robot, direct
Ethernet link (no go2_robot.py, no Tailscale/Zenoh).

Inputs:
  - RGB-D: plain TCP from realsense_stream_server.py on the Jetson
    (192.168.123.18:6000) -- identical client code to go2_robot.py /
    vlfm_navigator_node.py's _RealSenseStreamClient.
  - Pose: /state_estimation (GO2_STACK's own SLAM, arise_slam_mid360),
    relayed over local UDP by pose_udp_relay.py -- see that script's
    docstring for why (this env's rclpy doesn't work here; vlfm's own
    ObstacleMap needs frontier_exploration+numba+torch1.12, which only
    exists in the `vlfm` conda env, and that env's rclpy build is broken).

Run pose_udp_relay.py FIRST (separate terminal, ROS-sourced shell):
    source /opt/ros/jazzy/setup.bash
    python3 src/vlfm_bridge/scripts/pose_udp_relay.py

Then this script, in the `vlfm` conda env:
    cd /home/tommy/Taowen/VLFM_Project/vlfm
    PYTHONPATH=$(pwd) /home/tommy/miniconda3/envs/vlfm/bin/python \\
        /home/tommy/Taowen/GO2_STACK/src/vlfm_bridge/scripts/verify_native_mapping.py \\
        --target chair --max-seconds 60

Assumes the four VLM Flask servers (launch_vlfm_servers.sh) are already up.

Ctrl-C stops; saves a final map snapshot either way.
"""
import argparse
import os
import socket
import struct
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.utils.geometry_utils import rho_theta
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.grounding_dino import GroundingDINOClient

JETSON_HOST = os.environ.get("GO2_JETSON_HOST", "192.168.3.18")
CAMERA_PORT = 6000
POSE_UDP_PORT = 8765
WAYPOINT_UDP_PORT = 8766
WAYPOINT_MIN_REPUBLISH_DELTA_M = 0.05  # avoid spamming an unchanged goal

# -- Camera extrinsics, identical to go2_robot.py's measured values (same
# physical D435i mount on the same Go2) --------------------------------
_CAMERA_FORWARD_OFFSET_M = 0.15
# Re-measured 2026-09-17 -- see run_vlfm_pipeline.py's comment for the
# floor-height-vs-distance regression that found this; camera mount
# changed since the original go2_robot.py-era calibration.
_CAMERA_PITCH_UP_DEG = 0.0

# -- Mapping params, identical to my_tests/test_go2_pipeline_continuous.py --
MIN_DEPTH = 0.3
MAX_DEPTH = 3.0
CAMERA_HEIGHT_M = 0.44  # re-measured 2026-09-17, see _CAMERA_PITCH_UP_DEG comment above
MIN_HEIGHT = -CAMERA_HEIGHT_M + 0.10
MAX_HEIGHT = 0.6
MAP_SIZE = 800  # 40m x 40m at 20px/m
PIXELS_PER_METER = 20
FRONTIER_SEARCH_RADIUS = 1.0
DINO_CONF_THRESHOLD = 0.35
MAX_BOX_AREA_FRAC = 0.7
EDGE_CROP_FRAC = 0.08
LOOP_INTERVAL_S = 1.0

RES_DIR = "/home/tommy/Taowen/RES"
DINO_SYNONYMS = {
    "computer": "computer . computer tower . PC . monitor . gaming pc .",
    "fan": "fan . fans . ceiling fan . electric fan . cooling fan .",
    "chair": "chair . office chair . seat .",
    "person": "person . human .",
}


def _build_camera_to_body_transform() -> np.ndarray:
    theta = np.deg2rad(_CAMERA_PITCH_UP_DEG)
    c, s = np.cos(theta), np.sin(theta)
    transform = np.eye(4)
    transform[0, 0], transform[0, 2] = c, -s
    transform[2, 0], transform[2, 2] = s, c
    transform[0, 3] = _CAMERA_FORWARD_OFFSET_M
    return transform


_CAMERA_TO_BODY = _build_camera_to_body_transform()


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class WaypointUdpSender:
    """Sends (x, y, z) to waypoint_udp_relay.py, which republishes it as a
    geometry_msgs/PointStamped on /way_point for GO2_STACK's localPlanner.
    Publish-only: does not touch /joy or autonomyMode, does not start
    pathFollower -- see waypoint_udp_relay.py's docstring for the safety
    reasoning. This class only ever sends; it never causes robot motion by
    itself (that requires pathFollower to be separately running)."""

    FMT = "<3d"

    def __init__(self, port: int = WAYPOINT_UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)

    def send(self, x: float, y: float, z: float) -> None:
        self._sock.sendto(struct.pack(self.FMT, x, y, z), self._dest)


class PoseUdpClient:
    """Background listener for pose_udp_relay.py's packets -- same
    keep-latest-value threading pattern as _RealSenseStreamClient below."""

    FMT = "<8d"
    SIZE = struct.calcsize(FMT)

    def __init__(self, port: int = POSE_UDP_PORT):
        self._lock = threading.Lock()
        self._xy_yaw = None
        self._last_stamp = 0.0
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
            stamp, x, y, _z, qx, qy, qz, qw = struct.unpack(self.FMT, data)
            yaw = _quat_to_yaw(qx, qy, qz, qw)
            with self._lock:
                self._xy_yaw = (np.array([x, y]), yaw)
                self._last_stamp = stamp

    def xy_yaw(self):
        with self._lock:
            return self._xy_yaw

    def close(self) -> None:
        self._stop.set()


class _RealSenseStreamClient:
    """Copied from go2_robot.py -- see that file for the protocol details."""

    INTRINSICS_FMT = ">4fII f"
    INTRINSICS_SIZE = struct.calcsize(INTRINSICS_FMT)

    def __init__(self, host: str, port: int = 6000):
        self._host = host
        self._port = port
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
                        self._intrinsics = {
                            "fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy,
                            "width": width, "height": height,
                            "depth_scale": depth_scale,
                        }

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


def get_camera_transform(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    body_to_episodic = np.eye(4)
    body_to_episodic[0, 0], body_to_episodic[0, 1] = c, -s
    body_to_episodic[1, 0], body_to_episodic[1, 1] = s, c
    body_to_episodic[0, 3], body_to_episodic[1, 3] = xy[0], xy[1]
    return body_to_episodic @ _CAMERA_TO_BODY


def next_run_dir(base_dir: str) -> str:
    path = os.path.join(base_dir, "native_mapping_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    for sub in ("frames", "occupancy_map", "value_map", "depth"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="chair")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--pose-port", type=int, default=POSE_UDP_PORT)
    parser.add_argument("--waypoint-port", type=int, default=WAYPOINT_UDP_PORT)
    parser.add_argument(
        "--publish-waypoint", action=argparse.BooleanOptionalAction, default=True,
        help="Publish the best frontier to /way_point via waypoint_udp_relay.py "
             "(publish-only -- see that script's docstring; does not by itself "
             "cause the robot to move, since pathFollower must be separately running).",
    )
    args = parser.parse_args()

    target = args.target
    blip_query = f"Seems like there is a {target} ahead."
    dino_caption = DINO_SYNONYMS.get(target, f"{target} .")

    run_dir = next_run_dir(RES_DIR)
    log_path = os.path.join(run_dir, "log.txt")
    log_f = open(log_path, "a")

    def log(msg: str) -> None:
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"===== native-mapping verification run: {run_dir} =====")
    log(f"target={target!r} blip_query={blip_query!r} dino_caption={dino_caption!r}")

    blip2itm = BLIP2ITMClient(port=12182)
    grounding_dino = GroundingDINOClient(port=12181)

    log("Connecting to D435i TCP stream + pose UDP relay...")
    camera = _RealSenseStreamClient(JETSON_HOST, CAMERA_PORT)
    pose_client = PoseUdpClient(args.pose_port)
    waypoint_sender = WaypointUdpSender(args.waypoint_port) if args.publish_waypoint else None
    last_way_point = None
    if waypoint_sender is not None:
        log(f"Waypoint publishing ON -> udp://127.0.0.1:{args.waypoint_port} -> /way_point "
            "(requires waypoint_udp_relay.py running; publish-only, no motion by itself).")
    else:
        log("Waypoint publishing OFF (--no-publish-waypoint).")

    intr = camera.intrinsics()
    wait_start = time.time()
    while intr is None:
        if time.time() - wait_start > 30.0:
            raise RuntimeError("No camera intrinsics received within 30s -- is realsense_stream_server.py running on the Jetson?")
        time.sleep(0.2)
        intr = camera.intrinsics()
    fx, fy, width = intr["fx"], intr["fy"], intr["width"]
    fov = 2 * np.arctan(width / (2 * fx))
    log(f"Camera intrinsics OK: fx={fx:.1f} fy={fy:.1f} width={width} fov={np.rad2deg(fov):.1f}deg")

    wait_start = time.time()
    while pose_client.xy_yaw() is None:
        if time.time() - wait_start > 30.0:
            raise RuntimeError(
                "No pose received within 30s -- is pose_udp_relay.py running "
                "(source /opt/ros/jazzy/setup.bash first) and is /state_estimation publishing?"
            )
        time.sleep(0.2)
    log("Pose relay OK, first sample received.")

    obstacle_map = ObstacleMap(
        min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, agent_radius=0.2,
        area_thresh=0.5, hole_area_thresh=-1, size=MAP_SIZE, pixels_per_meter=PIXELS_PER_METER,
    )
    value_map = ValueMap(value_channels=1, size=MAP_SIZE, obstacle_map=obstacle_map)

    frame_idx = 0
    found_count = 0
    start_time = time.time()
    log(f"Starting loop (~{LOOP_INTERVAL_S}s/tick, Ctrl-C to stop)...")
    try:
        while time.time() - start_time < args.max_seconds:
            loop_t0 = time.time()
            frame_idx += 1

            xy_yaw = pose_client.xy_yaw()
            color, depth_raw = camera.latest()
            if xy_yaw is None or color is None or depth_raw is None:
                frame_idx -= 1
                log("  [warn] pose or camera frame not ready yet, skipping tick")
                time.sleep(LOOP_INTERVAL_S)
                continue
            xy, yaw = xy_yaw
            depth_norm = normalize_depth(depth_raw, intr["depth_scale"])
            tf_camera_to_episodic = get_camera_transform(xy, yaw)

            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(run_dir, "frames", f"tick_{frame_idx:05d}.jpg"), color_bgr)
            cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{frame_idx:05d}.png"), depth_raw)
            depth_vis = cv2.applyColorMap(
                cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                cv2.COLORMAP_JET,
            )
            depth_vis[depth_raw == 0] = (0, 0, 0)
            cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{frame_idx:05d}_vis.jpg"), depth_vis)

            try:
                detections = grounding_dino.predict(color, caption=dino_caption)
                valid_idxs = [
                    i for i, (box, logit) in enumerate(zip(detections.boxes, detections.logits))
                    if logit >= DINO_CONF_THRESHOLD and box_area_frac(box) <= MAX_BOX_AREA_FRAC
                ]
            except Exception as e:
                log(f"  [warn] GroundingDINO call failed: {e}")
                valid_idxs = []
            target_seen = len(valid_idxs) > 0

            try:
                score = blip2itm.cosine(color, blip_query)
            except Exception as e:
                log(f"  [warn] BLIP2ITM call failed: {e}")
                score = 0.0

            try:
                obstacle_map.update_map(
                    depth=depth_norm, tf_camera_to_episodic=tf_camera_to_episodic,
                    min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, fx=fx, fy=fy,
                    topdown_fov=fov, explore=True, update_obstacles=True,
                )
                obstacle_map.update_agent_traj(xy, yaw)
                value_map.update_map(
                    values=np.array([score]), depth=depth_norm, tf_camera_to_episodic=tf_camera_to_episodic,
                    min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, fov=fov,
                )
                value_map.update_agent_traj(xy, yaw)
            except IndexError as e:
                log(f"  [warn] pose fell outside map bounds, skipping map update: {e}")

            frontiers = obstacle_map.frontiers
            status = (
                f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s] "
                f"pose=({xy[0]:+.2f},{xy[1]:+.2f},yaw={yaw:+.2f}) score={score:+.3f} frontiers={len(frontiers)}"
            )
            if target_seen:
                found_count += 1
                best_idx = max(valid_idxs, key=lambda i: detections.logits[i])
                conf = detections.logits[best_idx]
                status += f"  >>> saw {target!r} conf={conf:.3f}"
            elif len(frontiers) > 0:
                sorted_frontiers, _ = value_map.sort_waypoints(frontiers, FRONTIER_SEARCH_RADIUS)
                best = sorted_frontiers[0]
                rho, theta = rho_theta(xy, yaw, best)
                status += f"  best_frontier=({best[0]:.2f},{best[1]:.2f}) turn={np.rad2deg(theta):+.0f}deg dist={rho:.2f}m"

                if waypoint_sender is not None:
                    moved = last_way_point is None or np.hypot(
                        best[0] - last_way_point[0], best[1] - last_way_point[1]
                    ) >= WAYPOINT_MIN_REPUBLISH_DELTA_M
                    if moved:
                        waypoint_sender.send(float(best[0]), float(best[1]), 0.0)
                        last_way_point = (float(best[0]), float(best[1]))
                        status += "  [/way_point sent]"
            log(status)

            cv2.imwrite(os.path.join(run_dir, "occupancy_map", f"tick_{frame_idx:05d}.png"), obstacle_map.visualize())
            cv2.imwrite(os.path.join(run_dir, "value_map", f"tick_{frame_idx:05d}.png"), value_map.visualize(obstacle_map=obstacle_map))

            elapsed = time.time() - loop_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)
    except KeyboardInterrupt:
        log("Stopped (Ctrl-C).")
    finally:
        camera.close()
        pose_client.close()
        cv2.imwrite(os.path.join(run_dir, "go2_obstacle_map_final.png"), obstacle_map.visualize())
        cv2.imwrite(os.path.join(run_dir, "go2_value_map_final.png"), value_map.visualize(obstacle_map=obstacle_map))
        log(f"Done. {frame_idx} ticks processed, target seen in {found_count}.")
        log(f"Results saved under: {run_dir}")
        log_f.close()


if __name__ == "__main__":
    main()
