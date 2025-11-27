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
    def __init__(self, host="0.0.0.0", port=5005, buffer_size=PACKET_SIZE):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.latest_packet = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Prepare socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))

        # Start background listener thread
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        """Background thread that constantly receives UDP packets."""
        while not self._stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                with self._lock:
                    self.latest_packet = (data, addr)  # store latest packet
            except OSError:
                break  # socket closed

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

    fmt = "<q" + "<?" + "8f" + "<?" + "8f"
    unpacked = struct.unpack(fmt, data)

    # --- Extract fields ---
    timestamp = unpacked[0]

    left_vals  = unpacked[1 : 1 + 1 + 8]     # bool + 8 floats
    right_vals = unpacked[1 + 1 + 8 : 1 + (1 + 8)*2]

    left_hand = HandPose(*left_vals)
    right_hand = HandPose(*right_vals)

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
        self.state_names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        self.socket: UdpListener | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{key}.pos": float for key in self.state_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.socket = UdpListener(host=self.config.host, port=self.config.port)

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
                if key != "enabled" or key != "gripper":
                    action["left_" + key] = val
        if right_hand.enabled:
            for key, val in right_hand.__dict__.items():
                if key != "enabled" or key != "gripper":
                    action["right_" + key] = val

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
