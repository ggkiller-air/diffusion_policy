import numpy as np
import pytest
import torch

from diffusion_policy.sonic.adapter import SonicPolicyAdapter, validate_observation
from diffusion_policy.sonic.config import SonicConfig, TactileMode
from diffusion_policy.sonic.dataset import (
    _cached_video_path,
    _valid_cached_video,
    assemble_action_78,
    assemble_state_46,
)
from diffusion_policy.sonic.policy import SonicDiffusionPolicy
from diffusion_policy.sonic.train import split_episode_indices


def small_config(mode: TactileMode) -> SonicConfig:
    return SonicConfig(
        mode=mode,
        image_height=16,
        image_width=16,
        vision_feature_dim=16,
        state_feature_dim=16,
        tactile_feature_dim=16,
        text_feature_dim=8,
        context_dim=32,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        num_diffusion_steps=4,
        num_inference_steps=2,
    )


def make_batch(mode: TactileMode) -> dict[str, torch.Tensor]:
    batch = {
        "state": torch.randn(1, 46),
        "images": torch.rand(1, 2, 3, 16, 16),
        "prompt_tokens": torch.ones(1, 128, dtype=torch.long),
        "actions": torch.randn(1, 40, 78),
    }
    if mode.uses_tactile:
        batch["tactile"] = torch.zeros(1, 768, dtype=torch.uint8)
        batch["future_tactile"] = torch.zeros(1, 4, 768, dtype=torch.uint8)
    if mode.dreams_state_and_vision:
        batch["future_state"] = torch.randn(1, 4, 46)
        batch["future_images"] = torch.rand(1, 4, 2, 3, 16, 16)
    return batch


def test_three_modes_have_exact_auxiliary_losses():
    expected = {
        TactileMode.NOTACTILE: {"action_loss", "loss"},
        TactileMode.HTD: {"action_loss", "tactile_jepa_loss", "loss"},
        TactileMode.JEPA: {
            "action_loss",
            "tactile_jepa_loss",
            "state_jepa_loss",
            "vision_jepa_loss",
            "loss",
        },
    }
    for mode in TactileMode:
        policy = SonicDiffusionPolicy(small_config(mode))
        assert set(policy(make_batch(mode))) == expected[mode]


def test_dream_horizon_must_fit_the_episode_safe_action_window():
    with pytest.raises(ValueError, match="smaller than action_horizon"):
        SonicConfig(dream_horizon=40)


def test_future_targets_do_not_change_action_context():
    policy = SonicDiffusionPolicy(small_config(TactileMode.JEPA)).eval()
    batch = make_batch(TactileMode.JEPA)
    with torch.no_grad():
        first = policy.encode_context(batch)
        batch["future_state"].normal_(mean=1000, std=10)
        batch["future_images"].fill_(1)
        batch["future_tactile"].fill_(255)
        second = policy.encode_context(batch)
    torch.testing.assert_close(first, second)


def test_dataset_assembly_matches_sonic_layout():
    raw = np.arange(43, dtype=np.float32)
    spans = {
        "left_leg": (0, 6),
        "right_leg": (6, 12),
        "waist": (12, 15),
        "left_arm": (15, 22),
        "left_hand": (22, 29),
        "right_arm": (29, 36),
        "right_hand": (36, 43),
    }
    state = assemble_state_46(raw, np.array([100, 101, 102]), spans)
    np.testing.assert_array_equal(state[22:29], raw[29:36])
    np.testing.assert_array_equal(state[29:36], raw[22:29])
    np.testing.assert_array_equal(state[-3:], [100, 101, 102])
    action = assemble_action_78(np.zeros(64), np.ones(7), np.full(7, 2))
    np.testing.assert_array_equal(action[64:71], 1)
    np.testing.assert_array_equal(action[71:78], 2)


def test_video_cache_validation_and_path(tmp_path):
    path = _cached_video_path(tmp_path, "observation.images.ego_view_left", 7)
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((3, 16, 20, 3), dtype=np.uint8), allow_pickle=False)
    assert _valid_cached_video(path, 3, 16, 20)
    assert not _valid_cached_video(path, 4, 16, 20)
    assert not _valid_cached_video(path, 3, 20, 16)

    np.save(path, np.zeros((3, 16, 20, 3), dtype=np.float32), allow_pickle=False)
    assert not _valid_cached_video(path, 3, 16, 20)


def test_episode_split_is_deterministic_and_disjoint(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        "".join(
            f'{{"episode_index": {episode}, "length": 100}}\n'
            for episode in range(20)
        ),
        encoding="utf-8",
    )
    train_a, val_a = split_episode_indices(tmp_path, 0.05, 42)
    train_b, val_b = split_episode_indices(tmp_path, 0.05, 42)

    assert (train_a, val_a) == (train_b, val_b)
    assert train_a.isdisjoint(val_a)
    assert train_a | val_a == set(range(20))
    assert len(val_a) == 1


def test_wire_validation_is_strict():
    observation = {
        "state": np.zeros(46, dtype=np.float32),
        "ego_view_left": np.zeros((16, 16, 3), dtype=np.uint8),
        "ego_view_right": np.zeros((16, 16, 3), dtype=np.uint8),
        "prompt": "move the bucket",
    }
    validate_observation(observation, requires_tactile=False)
    with pytest.raises(ValueError, match="requires tactile"):
        validate_observation(observation, requires_tactile=True)
    observation["tactile"] = np.zeros(768, dtype=np.uint8)
    validate_observation(observation, requires_tactile=True)


def test_adapter_returns_canonical_action_shape():
    policy = SonicDiffusionPolicy(small_config(TactileMode.NOTACTILE)).eval()
    adapter = SonicPolicyAdapter(policy, "cpu")
    result = adapter.infer(
        {
            "state": np.zeros(46, dtype=np.float32),
            "ego_view_left": np.zeros((16, 16, 3), dtype=np.uint8),
            "ego_view_right": np.zeros((16, 16, 3), dtype=np.uint8),
            "prompt": "move the bucket",
        }
    )
    assert adapter.metadata["protocol"] == "sonic_vla_v1"
    assert adapter.metadata["action_layout"]["motion_token"] == [0, 64]
    assert result["actions"].shape == (40, 78)
    assert result["actions"].dtype == np.float32
