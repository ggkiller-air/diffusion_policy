from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any


class TactileMode(str, enum.Enum):
    """The three Table 1 experiment modes."""

    NOTACTILE = "notactile"
    HTD = "htd"
    JEPA = "jepa"

    @property
    def uses_tactile(self) -> bool:
        return self is not TactileMode.NOTACTILE

    @property
    def dreams_tactile(self) -> bool:
        return self in (TactileMode.HTD, TactileMode.JEPA)

    @property
    def dreams_state_and_vision(self) -> bool:
        return self is TactileMode.JEPA


@dataclasses.dataclass(frozen=True)
class SonicConfig:
    mode: TactileMode = TactileMode.NOTACTILE
    image_height: int = 128
    image_width: int = 128
    state_dim: int = 46
    tactile_dim: int = 256
    action_dim: int = 78
    action_horizon: int = 40
    dream_horizon: int = 4
    vision_feature_dim: int = 128
    state_feature_dim: int = 128
    tactile_feature_dim: int = 128
    text_feature_dim: int = 64
    context_dim: int = 256
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (128, 256, 512)
    num_diffusion_steps: int = 100
    num_inference_steps: int = 10
    jepa_loss_weight: float = 0.1
    teacher_momentum: float = 0.99
    max_prompt_bytes: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", TactileMode(self.mode))
        if isinstance(self.down_dims, list):
            object.__setattr__(self, "down_dims", tuple(self.down_dims))
        expected = {
            "state_dim": 46,
            "tactile_dim": 256,
            "action_dim": 78,
            "action_horizon": 40,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"sonic_vla_v1 requires {name}={value}, got {getattr(self, name)}"
                )
        if self.dream_horizon < 1:
            raise ValueError("dream_horizon must be positive")
        if self.dream_horizon >= self.action_horizon:
            raise ValueError("dream_horizon must be smaller than action_horizon")
        if self.image_height < 16 or self.image_width < 16:
            raise ValueError("image dimensions must be at least 16 pixels")
        if not 0.0 <= self.teacher_momentum < 1.0:
            raise ValueError("teacher_momentum must be in [0, 1)")
        if not self.down_dims or any(dim % 8 for dim in self.down_dims):
            raise ValueError("down_dims must be non-empty and divisible by 8")

    @classmethod
    def from_json(cls, path: str | Path) -> SonicConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SonicConfig:
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = set(values) - fields
        if unknown:
            raise ValueError(f"unknown SonicConfig keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["mode"] = self.mode.value
        result["down_dims"] = list(self.down_dims)
        return result


PROTOCOL = "sonic_vla_v1"
STATE_DIM = 46
TACTILE_DIM = 256
ACTION_DIM = 78
ACTION_HORIZON = 40
VIDEO_KEYS = ("ego_view_left", "ego_view_right")
STATE_GROUP_ORDER = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)
ACTION_LAYOUT = {
    "motion_token": [0, 64],
    "left_hand_joints": [64, 71],
    "right_hand_joints": [71, 78],
}


def tokenize_prompt(prompt: str, max_bytes: int = 128):
    """Encode UTF-8 bytes as stable token IDs; 0 is reserved for padding."""
    import numpy as np

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    result = np.zeros(max_bytes, dtype=np.int64)
    encoded = prompt.encode("utf-8")[:max_bytes]
    if encoded:
        result[: len(encoded)] = (
            np.frombuffer(encoded, dtype=np.uint8).astype(np.int64) + 1
        )
    return result
