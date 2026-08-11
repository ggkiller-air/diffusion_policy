from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.sonic.config import SonicConfig


class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PromptEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, output_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled)


class SonicDiffusionPolicy(nn.Module):
    """Stereo diffusion policy with optional HTD/UniVLaT auxiliary prediction.

    Future observations are consumed only by frozen teacher encoders in
    ``compute_losses``. ``encode_context`` and action inference accept current
    observations only, which structurally prevents future-label leakage.
    """

    def __init__(self, config: SonicConfig) -> None:
        super().__init__()
        self.config = config
        self.left_vision_encoder = VisionEncoder(config.vision_feature_dim)
        self.right_vision_encoder = VisionEncoder(config.vision_feature_dim)
        self.state_encoder = VectorEncoder(config.state_dim, config.state_feature_dim)
        self.prompt_encoder = PromptEncoder(config.text_feature_dim)
        context_input_dim = (
            config.vision_feature_dim * 2
            + config.state_feature_dim
            + config.text_feature_dim
        )
        if config.mode.uses_tactile:
            self.tactile_encoder = VectorEncoder(
                config.tactile_dim, config.tactile_feature_dim
            )
            context_input_dim += config.tactile_feature_dim
        else:
            self.tactile_encoder = None
        self.context_projection = nn.Sequential(
            nn.Linear(context_input_dim, config.context_dim),
            nn.LayerNorm(config.context_dim),
            nn.SiLU(),
        )
        self.diffusion_model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.context_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))

        self.teacher_state_encoder = None
        self.teacher_left_vision_encoder = None
        self.teacher_right_vision_encoder = None
        self.teacher_tactile_encoder = None
        self.state_predictor = None
        self.vision_predictor = None
        self.tactile_predictor = None
        if config.mode.dreams_tactile:
            self.teacher_tactile_encoder = self._teacher_copy(self.tactile_encoder)
            self.tactile_predictor = nn.Linear(
                config.context_dim, config.dream_horizon * config.tactile_feature_dim
            )
        if config.mode.dreams_state_and_vision:
            self.teacher_state_encoder = self._teacher_copy(self.state_encoder)
            self.teacher_left_vision_encoder = self._teacher_copy(
                self.left_vision_encoder
            )
            self.teacher_right_vision_encoder = self._teacher_copy(
                self.right_vision_encoder
            )
            self.state_predictor = nn.Linear(
                config.context_dim, config.dream_horizon * config.state_feature_dim
            )
            self.vision_predictor = nn.Linear(
                config.context_dim,
                config.dream_horizon * 2 * config.vision_feature_dim,
            )

    @staticmethod
    def _teacher_copy(module: nn.Module | None) -> nn.Module:
        if module is None:
            raise ValueError("cannot create a teacher without an online encoder")
        teacher = copy.deepcopy(module)
        teacher.requires_grad_(False)
        teacher.eval()
        return teacher

    def train(self, mode: bool = True):
        super().train(mode)
        for teacher in self._teacher_modules():
            teacher.eval()
        return self

    def _teacher_modules(self) -> list[nn.Module]:
        return [
            module
            for module in (
                self.teacher_state_encoder,
                self.teacher_left_vision_encoder,
                self.teacher_right_vision_encoder,
                self.teacher_tactile_encoder,
            )
            if module is not None
        ]

    @torch.no_grad()
    def set_normalization_stats(self, stats: Mapping[str, torch.Tensor]) -> None:
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            value = torch.as_tensor(stats[name], dtype=getattr(self, name).dtype)
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"normalization {name} has shape {value.shape}")
            if name.endswith("_std") and torch.any(value <= 0):
                raise ValueError(f"normalization {name} must be positive")
            getattr(self, name).copy_(value)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state.float() - self.state_mean) / self.state_std

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action.float() - self.action_mean) / self.action_std

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

    def encode_context(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        state = batch["state"]
        images = batch["images"]
        prompt_tokens = batch["prompt_tokens"]
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must be [B, {self.config.state_dim}], got {state.shape}"
            )
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError(f"images must be [B, 2, 3, H, W], got {images.shape}")
        features = [
            self.left_vision_encoder(images[:, 0].float()),
            self.right_vision_encoder(images[:, 1].float()),
            self.state_encoder(self._normalize_state(state)),
            self.prompt_encoder(prompt_tokens.long()),
        ]
        if self.config.mode.uses_tactile:
            tactile = batch.get("tactile")
            if (
                tactile is None
                or tactile.ndim != 2
                or tactile.shape[-1] != self.config.tactile_dim
            ):
                raise ValueError(
                    f"{self.config.mode.value} requires tactile [B, {self.config.tactile_dim}]"
                )
            features.append(self.tactile_encoder(tactile.float().div(255.0)))
        return self.context_projection(torch.cat(features, dim=-1))

    @staticmethod
    def _latent_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target.detach(), dim=-1)
        return F.smooth_l1_loss(prediction, target)

    def _auxiliary_losses(
        self, context: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses = {}
        batch_size = context.shape[0]
        horizon = self.config.dream_horizon
        with torch.no_grad():
            if self.config.mode.dreams_tactile:
                future_tactile = batch.get("future_tactile")
                if future_tactile is None or future_tactile.shape[1:] != (
                    horizon,
                    self.config.tactile_dim,
                ):
                    raise ValueError(
                        f"HTD/JEPA requires future_tactile [B, H, {self.config.tactile_dim}]"
                    )
                tactile_target = self.teacher_tactile_encoder(
                    future_tactile.float()
                    .div(255.0)
                    .reshape(-1, self.config.tactile_dim)
                ).reshape(batch_size, horizon, self.config.tactile_feature_dim)
            if self.config.mode.dreams_state_and_vision:
                future_state = batch.get("future_state")
                future_images = batch.get("future_images")
                if future_state is None or future_state.shape[1:] != (
                    horizon,
                    self.config.state_dim,
                ):
                    raise ValueError("JEPA requires future_state [B, H, 46]")
                if (
                    future_images is None
                    or future_images.ndim != 6
                    or future_images.shape[1:3]
                    != (
                        horizon,
                        2,
                    )
                ):
                    raise ValueError("JEPA requires future_images [B, H, 2, 3, H, W]")
                state_target = self.teacher_state_encoder(
                    self._normalize_state(future_state).reshape(
                        -1, self.config.state_dim
                    )
                ).reshape(batch_size, horizon, self.config.state_feature_dim)
                left_target = self.teacher_left_vision_encoder(
                    future_images[:, :, 0].reshape(-1, *future_images.shape[3:]).float()
                ).reshape(batch_size, horizon, self.config.vision_feature_dim)
                right_target = self.teacher_right_vision_encoder(
                    future_images[:, :, 1].reshape(-1, *future_images.shape[3:]).float()
                ).reshape(batch_size, horizon, self.config.vision_feature_dim)
                vision_target = torch.stack((left_target, right_target), dim=2)
        if self.config.mode.dreams_tactile:
            tactile_prediction = self.tactile_predictor(context).reshape(
                batch_size, horizon, self.config.tactile_feature_dim
            )
            losses["tactile_jepa_loss"] = self._latent_loss(
                tactile_prediction, tactile_target
            )
        if self.config.mode.dreams_state_and_vision:
            state_prediction = self.state_predictor(context).reshape(
                batch_size, horizon, self.config.state_feature_dim
            )
            vision_prediction = self.vision_predictor(context).reshape(
                batch_size, horizon, 2, self.config.vision_feature_dim
            )
            losses["state_jepa_loss"] = self._latent_loss(
                state_prediction, state_target
            )
            losses["vision_jepa_loss"] = self._latent_loss(
                vision_prediction, vision_target
            )
        return losses

    def compute_losses(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        context = self.encode_context(batch)
        actions = batch["actions"]
        if actions.shape[1:] != (self.config.action_horizon, self.config.action_dim):
            raise ValueError("actions must be [B, 40, 78]")
        normalized_actions = self._normalize_action(actions)
        noise = torch.randn_like(normalized_actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        )
        noisy_actions = self.noise_scheduler.add_noise(
            normalized_actions, noise, timesteps
        )
        predicted_noise = self.diffusion_model(
            noisy_actions, timesteps, global_cond=context
        )
        action_loss = F.mse_loss(predicted_noise, noise)
        result = {"action_loss": action_loss}
        result.update(self._auxiliary_losses(context, batch))
        auxiliary = sum(result.values()) - action_loss
        result["loss"] = action_loss + self.config.jepa_loss_weight * auxiliary
        return result

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.compute_losses(batch)

    @torch.no_grad()
    def update_teacher(self) -> None:
        momentum = self.config.teacher_momentum
        pairs = (
            (self.state_encoder, self.teacher_state_encoder),
            (self.left_vision_encoder, self.teacher_left_vision_encoder),
            (self.right_vision_encoder, self.teacher_right_vision_encoder),
            (self.tactile_encoder, self.teacher_tactile_encoder),
        )
        for online, teacher in pairs:
            if online is None or teacher is None:
                continue
            for online_parameter, teacher_parameter in zip(
                online.parameters(), teacher.parameters()
            ):
                teacher_parameter.mul_(momentum).add_(
                    online_parameter, alpha=1.0 - momentum
                )

    @torch.no_grad()
    def predict_actions(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        context = self.encode_context(batch)
        sample = torch.randn(
            context.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            self.config.num_inference_steps, device=context.device
        )
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.diffusion_model(sample, timestep, global_cond=context)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        result = self._unnormalize_action(sample).float()
        if not torch.isfinite(result).all():
            raise RuntimeError("diffusion policy produced NaN or infinity")
        return result
