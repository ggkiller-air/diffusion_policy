from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from diffusion_policy.sonic.config import SonicConfig
from diffusion_policy.sonic.policy import SonicDiffusionPolicy


def save_checkpoint(
    path: str | Path,
    model: SonicDiffusionPolicy,
    *,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "sonic_diffusion_policy_v1",
        "step": int(step),
        "config": model.config.to_dict(),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint_payload(
    path: str | Path, device: str | torch.device = "cpu"
) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "sonic_diffusion_policy_v1":
        raise ValueError("not a sonic_diffusion_policy_v1 checkpoint")
    return payload


def load_policy(
    path: str | Path, device: str | torch.device = "cpu"
) -> SonicDiffusionPolicy:
    payload = load_checkpoint_payload(path, device)
    policy = SonicDiffusionPolicy(SonicConfig.from_dict(payload["config"]))
    policy.load_state_dict(payload["model"], strict=True)
    policy.to(device).eval()
    return policy
