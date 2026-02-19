import logging
import socket
import struct
import threading
from dataclasses import dataclass

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_varjo_xr3 import VarjoXR3Config

logger = logging.getLogger(__name__)

PACKET_SIZE = 74


class UdpListener:
    def __init__(self, remote_address: str, port=5005, local_address="0.0.0.0", buffer_size=PACKET_SIZE):
        self.local_address = local_address
        self.port = port
        self.buffer_size = buffer_size
        self.latest_packet = None
        self.running = False
        self.remote_address = remote_address
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Prepare socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((local_address, port))

        # Start background listener thread
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        """Background thread that constantly receives UDP packets."""
        self.running = True
        while not self._stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                if addr[0] != self.remote_address:
                    continue
                with self._lock:
                    self.latest_packet = (data, addr)  # store latest packet
            except OSError:
                break  # socket closed
        self.running = False

    @property
    def is_connected(self) -> bool:
        return self.running

    def get_latest_packet(self):
        """Thread-safe read access."""
        with self._lock:
            return self.latest_packet

    def stop(self):
        """Clean shutdown."""
        self._stop_event.set()
        self.sock.close()
        self.thread.join()


@dataclass
class HandPose:
    enabled: bool
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    gripper: float


def _parse_hand_data(data: bytes):
    """
    Layout:
    <q      timestamp   (8 bytes)
    <?f f f f f f f f   left hand  (1 bool + 8 floats)
    <?f f f f f f f f   right hand (same)
    """

    unpacked = struct.unpack("<q?ffffffff?ffffffff", data)

    # --- Extract fields ---
    timestamp = unpacked[0]

    left_vals = unpacked[1 : 1 + 1 + 8]  # bool + 8 floats
    right_vals = unpacked[1 + 1 + 8 : 1 + (1 + 8) * 2]

    # Pack in to hand pose object with mm as the unit
    left_hand = HandPose(*left_vals)
    left_hand.x *= 1000.0
    left_hand.y *= 1000.0
    left_hand.z *= 1000.0
    right_hand = HandPose(*right_vals)
    right_hand.x *= 1000.0
    right_hand.y *= 1000.0
    right_hand.z *= 1000.0

    return timestamp, left_hand, right_hand


class VarjoXR3(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = VarjoXR3Config
    name = "varjo_xr3"

    def __init__(self, config: VarjoXR3Config):
        super().__init__(config)
        self.config = config
        self.state_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
        self.socket: UdpListener | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{key}.pos": float for key in self.state_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return bool(self.socket is not None and self.socket.is_connected)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.socket = UdpListener(remote_address=self.config.host, port=self.config.port)

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        if self.socket is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        data = self.socket.get_latest_packet()

        if data is None:
            print("Warning: No packet received from client")
            return {}

        _, left_hand, right_hand = _parse_hand_data(data[0])

        action = {}
        if left_hand.enabled:
            for key, val in left_hand.__dict__.items():
                if key == "enabled":
                    continue
                action["left_" + key + ".pos"] = val
        if right_hand.enabled:
            for key, val in right_hand.__dict__.items():
                if key == "enabled":
                    continue
                action["right_" + key + ".pos"] = val

        # if not (left_hand.enabled and right_hand.enabled):
        #     print("No tracking data")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")
        if self.socket is not None:
            self.socket.stop()
        logger.info(f"{self} disconnected.")
