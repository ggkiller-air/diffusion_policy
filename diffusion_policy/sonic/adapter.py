from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.sonic.config import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_LAYOUT,
    PROTOCOL,
    STATE_DIM,
    TACTILE_DIM,
    VIDEO_KEYS,
    tokenize_prompt,
)


def validate_observation(
    observation: Mapping[str, Any], *, requires_tactile: bool
) -> dict:
    state = np.asarray(observation.get("state"))
    if state.dtype != np.float32 or state.shape != (STATE_DIM,):
        raise ValueError(
            f"state must be float32[{STATE_DIM}], got {state.dtype} {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or infinity")
    result = {"state": state}
    image_shape = None
    for key in VIDEO_KEYS:
        image = np.asarray(observation.get(key))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"{key} must be uint8[H, W, 3], got {image.dtype} {image.shape}"
            )
        if image_shape is not None and image.shape != image_shape:
            raise ValueError("stereo images must have identical shapes")
        image_shape = image.shape
        result[key] = image
    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    result["prompt"] = prompt
    tactile_value = observation.get("tactile")
    if requires_tactile and tactile_value is None:
        raise ValueError(f"this checkpoint requires tactile uint8[{TACTILE_DIM}]")
    if tactile_value is not None:
        tactile = np.asarray(tactile_value)
        if tactile.dtype != np.uint8 or tactile.shape != (TACTILE_DIM,):
            raise ValueError(
                f"tactile must be uint8[{TACTILE_DIM}], got {tactile.dtype} {tactile.shape}"
            )
        result["tactile"] = tactile
    return result


class SonicPolicyAdapter:
    def __init__(self, policy, device: str | torch.device) -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.requires_tactile = policy.config.mode.uses_tactile

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "backend": "diffusion_policy",
            "state_dim": STATE_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "video_keys": list(VIDEO_KEYS),
            "requires_tactile": self.requires_tactile,
            "action_layout": ACTION_LAYOUT,
        }

    def _image_tensor(self, image: np.ndarray) -> torch.Tensor:
        value = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )
        return F.interpolate(
            value,
            size=(self.policy.config.image_height, self.policy.config.image_width),
            mode="bilinear",
            align_corners=False,
        )[0]

    @torch.inference_mode()
    def infer(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        obs = validate_observation(observation, requires_tactile=self.requires_tactile)
        batch = {
            "state": torch.from_numpy(obs["state"]).unsqueeze(0).to(self.device),
            "images": torch.stack(
                [self._image_tensor(obs[key]) for key in VIDEO_KEYS], dim=0
            )
            .unsqueeze(0)
            .to(self.device),
            "prompt_tokens": torch.from_numpy(
                tokenize_prompt(obs["prompt"], self.policy.config.max_prompt_bytes)
            )
            .unsqueeze(0)
            .to(self.device),
        }
        if self.requires_tactile:
            batch["tactile"] = (
                torch.from_numpy(obs["tactile"]).unsqueeze(0).to(self.device)
            )
        actions = self.policy.predict_actions(batch)[0].cpu().numpy().astype(np.float32)
        if (
            actions.shape != (ACTION_HORIZON, ACTION_DIM)
            or not np.isfinite(actions).all()
        ):
            raise RuntimeError(f"policy returned invalid actions {actions.shape}")
        return {"actions": actions}
