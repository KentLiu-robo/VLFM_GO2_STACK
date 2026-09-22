# GO2Robot: BaseRobot adapter for the Unitree GO2, talking to the real
# robot over the campus-WiFi + Tailscale + Zenoh bridge set up separately
# (see the "go2-tailscale-session" ops notes). No unitree_sdk2 (C++) code is
# used here -- this goes through the official Python SDK (unitree_sdk2py)
# instead, subscribing to the robot's own DDS topics through the same
# loopback domain the Zenoh bridge publishes into on this host.
#
# Camera: a single Intel RealSense D435i mounted on the robot's Jetson,
# streamed to this host over a small custom TCP protocol (see
# realsense_stream_server.py on the Jetson). GO2 has one camera, not Spot's
# multi-camera array, so this defines its own minimal camera-id scheme
# rather than reusing Spot's CAM_ID_TO_SHAPE.
#
# command_base_velocity() was verified end-to-end on hardware 2026-09-04
# (slow in-place rotation, vyaw=0.15 rad/s) after a rear-right leg joint
# issue was found and resolved via a robot power cycle.
import os
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base_robot import BaseRobot
from .frame_ids import SpotFrameIds

# --- CycloneDDS wiring -------------------------------------------------
# unitree_sdk2py always builds its own explicit CycloneDDS XML config (it
# does not fall back to the CYCLONEDDS_URI env var the way the C++ SDK's
# ChannelFactory::Init(0, "") does), so the interface-with-name template is
# patched here to add AllowMulticast=false + a unicast Peer to 127.0.0.1 --
# the same settings that make the host-side loopback Zenoh bridge
# discoverable. Without this, discovery tries multicast on "lo", which
# doesn't support it, and nothing is ever received.
import unitree_sdk2py.core.channel as _ch  # noqa: E402

_ch.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>
                </Interfaces>
                <AllowMulticast>false</AllowMulticast>
            </General>
            <Discovery>
                <Peers>
                    <Peer Address="127.0.0.1"/>
                </Peers>
            </Discovery>
        </Domain>
    </CycloneDDS>"""

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.go2.sport.sport_client import SportClient  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_  # noqa: E402

TOPIC_HIGHSTATE = "rt/sportmodestate"

# Camera extrinsics relative to the body-frame origin (the point reported by
# rt/sportmodestate's position field), measured by hand 2026-09-05:
#   - ~15cm forward of the body origin, on the centerline (no left/right offset)
#   - tilted up (pitch) ~10 degrees from level
#   - no yaw or roll offset (mounted square with the body, facing forward)
# Axes follow the x=forward, y=left, z=up convention used throughout
# get_point_cloud()/get_transform() -- NOT the raw camera optical frame.
_CAMERA_FORWARD_OFFSET_M = 0.15
# Refined from 10.0 via data (2026-09-05): checked whether the lowest ("floor")
# points' computed height stayed constant across depth bins in a live frame --
# it drifted upward with distance (+0.0246 m per meter), implying the pitch
# was overestimated by about 1.4 deg. See calibrate_pitch.py in scratchpad.
_CAMERA_PITCH_UP_DEG = 8.6


def _build_camera_to_body_transform() -> np.ndarray:
    theta = np.deg2rad(_CAMERA_PITCH_UP_DEG)
    c, s = np.cos(theta), np.sin(theta)
    # Pitch-up rotation about the left (y) axis: the camera's own forward
    # axis (+x) tilts toward +z (up) in the body frame by `theta`.
    transform = np.eye(4)
    transform[0, 0], transform[0, 2] = c, -s
    transform[2, 0], transform[2, 2] = s, c
    transform[0, 3] = _CAMERA_FORWARD_OFFSET_M
    return transform


_CAMERA_TO_BODY = _build_camera_to_body_transform()

_channel_factory_lock = threading.Lock()
_channel_factory_ready = False


def _ensure_channel_factory(network_interface: str = "lo") -> None:
    """ChannelFactoryInitialize is a process-wide singleton; only call it once."""
    global _channel_factory_ready
    with _channel_factory_lock:
        if not _channel_factory_ready:
            ChannelFactoryInitialize(0, network_interface)
            _channel_factory_ready = True


class GO2CamIds:
    """GO2 only has one onboard RealSense; no Spot-style camera array."""

    FRONT_COLOR = "front_color"
    FRONT_DEPTH = "front_depth"


CAM_ID_TO_SHAPE = {
    GO2CamIds.FRONT_COLOR: (480, 640, 3),
    GO2CamIds.FRONT_DEPTH: (480, 640, 1),
}


class _RealSenseStreamClient:
    """Background thread that keeps the latest RGB-D frame from the
    Jetson's realsense_stream_server.py, so get_camera_images() can return
    instantly instead of blocking on a fresh TCP round-trip each call.

    Also captures the depth camera's intrinsics (fx, fy, ppx, ppy,
    depth_scale), sent once by the server right after connecting -- needed
    for ObstacleMap.update_map()'s depth-to-point-cloud projection.
    """

    INTRINSICS_FMT = ">4fII f"
    INTRINSICS_SIZE = struct.calcsize(INTRINSICS_FMT)

    def __init__(self, host: str, port: int = 6000):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._intrinsics: Optional[dict] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
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
                time.sleep(1.0)  # reconnect backoff

    def intrinsics(self) -> Optional[dict]:
        with self._lock:
            return self._intrinsics

    def latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self._color, self._depth

    def close(self) -> None:
        self._stop.set()


class GO2Robot(BaseRobot):
    def __init__(self, jetson_host: str = os.environ.get("GO2_JETSON_HOST", "192.168.3.18"), camera_port: int = 6000):
        _ensure_channel_factory("lo")

        self._latest_state: Optional[SportModeState_] = None
        self._state_lock = threading.Lock()
        self._suber = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
        self._suber.Init(self._state_handler, 1)

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self._camera = _RealSenseStreamClient(jetson_host, camera_port)

        # Wait briefly for the first pose message so xy_yaw is valid immediately.
        for _ in range(50):
            if self._latest_state is not None:
                break
            time.sleep(0.1)

    def _state_handler(self, msg: SportModeState_) -> None:
        with self._state_lock:
            self._latest_state = msg

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        with self._state_lock:
            state = self._latest_state
        if state is None:
            raise RuntimeError("No SportModeState_ received yet from rt/sportmodestate")
        xy = np.array([state.position[0], state.position[1]])
        yaw = state.imu_state.rpy[2]
        return xy, yaw

    @property
    def arm_joints(self) -> np.ndarray:
        # GO2 has no arm.
        return np.zeros(6)

    @property
    def camera_intrinsics(self) -> Optional[dict]:
        """fx, fy, ppx, ppy, width, height, depth_scale from the RealSense
        depth stream, or None if not received yet."""
        return self._camera.intrinsics()

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        for source in camera_source:
            assert source in CAM_ID_TO_SHAPE, f"Invalid camera source: {source}"
        color, depth = self._camera.latest()
        images = {}
        for source in camera_source:
            if source == GO2CamIds.FRONT_COLOR:
                if color is None:
                    raise RuntimeError("No color frame received yet from RealSense stream")
                images[source] = color
            elif source == GO2CamIds.FRONT_DEPTH:
                if depth is None:
                    raise RuntimeError("No depth frame received yet from RealSense stream")
                images[source] = depth
        return images

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        # NOT YET RE-VERIFIED ON HARDWARE after the rear-right leg issue --
        # do not call until that is resolved.
        self.sport_client.Move(lin_vel, 0, ang_vel)

    def get_transform(self, frame: str = SpotFrameIds.BODY) -> np.ndarray:
        """Body-frame pose in the episodic frame (per BaseRobot's contract --
        does NOT include the camera extrinsics; see get_camera_transform())."""
        (x, y), yaw = self.xy_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        transform = np.eye(4)
        transform[0, 0], transform[0, 1] = c, -s
        transform[1, 0], transform[1, 1] = s, c
        transform[0, 3] = x
        transform[1, 3] = y
        return transform

    def get_camera_transform(self) -> np.ndarray:
        """Camera pose in the episodic frame: body pose composed with the
        fixed camera-to-body extrinsics (measured offset/tilt -- see
        _CAMERA_TO_BODY above). This is what mapping code (ObstacleMap,
        ValueMap) should pass as tf_camera_to_episodic, not get_transform()."""
        return self.get_transform() @ _CAMERA_TO_BODY

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        pass  # GO2 has no arm.

    def open_gripper(self) -> None:
        pass  # GO2 has no gripper.

    def close(self) -> None:
        self._camera.close()
