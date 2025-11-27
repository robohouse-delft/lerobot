from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("varjo_xr3")
@dataclass
class VarjoXR3Config(TeleoperatorConfig):
    host: str
    port: int
