from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from diffusion_policy.sonic.config import (
    ACTION_DIM,
    ACTION_HORIZON,
    STATE_DIM,
    STATE_GROUP_ORDER,
    TACTILE_DIM,
    SonicConfig,
    tokenize_prompt,
)

OBSERVATION_ACTION_COLUMNS = (
    "observation.state",
    "observation.projected_gravity",
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
)
TACTILE_COLUMNS = (
    "observation.tactile_vest",
    "observation.tactile_left_arm",
    "observation.tactile_right_arm",
)
PARQUET_COLUMNS = (
    *OBSERVATION_ACTION_COLUMNS,
    *TACTILE_COLUMNS,
    "task_index",
)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_state_spans(dataset_path: str | Path) -> dict[str, tuple[int, int]]:
    modality = _read_json(Path(dataset_path) / "meta" / "modality.json")
    state = modality.get("state", {})
    spans = {}
    for group in STATE_GROUP_ORDER:
        spec = state.get(group)
        if not spec or "start" not in spec or "end" not in spec:
            raise ValueError(f"meta/modality.json is missing state.{group}.start/end")
        spans[group] = (int(spec["start"]), int(spec["end"]))
    widths = [spans[name][1] - spans[name][0] for name in STATE_GROUP_ORDER]
    if widths != [6, 6, 3, 7, 7, 7, 7]:
        raise ValueError(f"unexpected SONIC state group widths: {widths}")
    return spans


def assemble_state_46(state_43, projected_gravity, spans) -> np.ndarray:
    state = np.asarray(state_43, dtype=np.float32)
    gravity = np.asarray(projected_gravity, dtype=np.float32)
    result = np.concatenate(
        [state[..., spans[group][0] : spans[group][1]] for group in STATE_GROUP_ORDER]
        + [gravity],
        axis=-1,
    )
    if result.shape[-1] != STATE_DIM:
        raise ValueError(
            f"canonical SONIC state must be {STATE_DIM}D, got {result.shape}"
        )
    return result


def assemble_action_78(motion_token, left_hand, right_hand) -> np.ndarray:
    result = np.concatenate(
        [
            np.asarray(motion_token, dtype=np.float32),
            np.asarray(left_hand, dtype=np.float32),
            np.asarray(right_hand, dtype=np.float32),
        ],
        axis=-1,
    )
    if result.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"canonical SONIC action must be {ACTION_DIM}D, got {result.shape}"
        )
    return result


def compute_normalization_stats(dataset_path: str | Path) -> dict[str, np.ndarray]:
    """Compute train/inference normalization without decoding video."""
    import pyarrow.parquet as pq

    root = Path(dataset_path)
    info = _read_json(root / "meta" / "info.json")
    spans = load_state_spans(root)
    state_sum = np.zeros(STATE_DIM, dtype=np.float64)
    state_sq_sum = np.zeros(STATE_DIM, dtype=np.float64)
    action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    action_sq_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    count = 0
    for episode in _read_jsonl(root / "meta" / "episodes.jsonl"):
        episode_index = int(episode["episode_index"])
        chunk = episode_index // int(info.get("chunks_size", 1000))
        path = root / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        table = pq.read_table(path, columns=list(OBSERVATION_ACTION_COLUMNS))
        state = assemble_state_46(
            np.asarray(table["observation.state"].to_pylist()),
            np.asarray(table["observation.projected_gravity"].to_pylist()),
            spans,
        ).astype(np.float64)
        action = assemble_action_78(
            np.asarray(table["action.motion_token"].to_pylist()),
            np.asarray(table["teleop.left_hand_joints"].to_pylist()),
            np.asarray(table["teleop.right_hand_joints"].to_pylist()),
        ).astype(np.float64)
        state_sum += state.sum(axis=0)
        state_sq_sum += np.square(state).sum(axis=0)
        action_sum += action.sum(axis=0)
        action_sq_sum += np.square(action).sum(axis=0)
        count += len(state)
    if count == 0:
        raise ValueError("SONIC dataset contains no frames")
    state_mean = state_sum / count
    action_mean = action_sum / count
    state_std = np.sqrt(np.maximum(state_sq_sum / count - np.square(state_mean), 1e-8))
    action_std = np.sqrt(
        np.maximum(action_sq_sum / count - np.square(action_mean), 1e-8)
    )
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
    }


class SonicLeRobotDataset(Dataset):
    """Episode-safe LeRobot loader for G1 stereo, tactile, and SONIC tokens."""

    def __init__(
        self,
        dataset_path: str | Path,
        config: SonicConfig,
        *,
        parquet_cache_size: int = 2,
    ) -> None:
        self.root = Path(dataset_path)
        self.config = config
        self.info = _read_json(self.root / "meta" / "info.json")
        self.fps = float(self.info["fps"])
        self.chunk_size = int(self.info.get("chunks_size", 1000))
        self.state_spans = load_state_spans(self.root)
        tasks = _read_jsonl(self.root / "meta" / "tasks.jsonl")
        self.tasks = {int(item["task_index"]): item["task"] for item in tasks}
        self.episodes = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.valid_lengths = []
        for episode in self.episodes:
            valid = int(episode["length"]) - config.action_horizon + 1
            self.valid_lengths.append(max(0, valid))
        self.cumulative_lengths = np.cumsum(self.valid_lengths).tolist()
        self._parquet_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._parquet_cache_size = parquet_cache_size
        if not self.cumulative_lengths or self.cumulative_lengths[-1] == 0:
            raise ValueError("dataset has no complete SONIC action windows")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect.bisect_right(self.cumulative_lengths, index)
        previous = (
            self.cumulative_lengths[episode_position - 1] if episode_position else 0
        )
        return episode_position, index - previous

    def _format_path(self, template: str, episode_index: int, **values) -> Path:
        return self.root / template.format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
            **values,
        )

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        cached = self._parquet_cache.pop(episode_index, None)
        if cached is not None:
            self._parquet_cache[episode_index] = cached
            return cached
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self._format_path(self.info["data_path"], episode_index)
        columns = [*OBSERVATION_ACTION_COLUMNS, "task_index"]
        if self.config.mode.uses_tactile:
            columns.extend(TACTILE_COLUMNS)
        table = pq.read_table(path, columns=columns)
        data = {name: np.asarray(table[name].to_pylist()) for name in columns}
        if self.config.mode.uses_tactile:
            streams = []
            for name in TACTILE_COLUMNS:
                column = table[name]
                if not pa.types.is_list(column.type) or not pa.types.is_uint8(
                    column.type.value_type
                ):
                    raise ValueError(f"{name} must be uint8, got {column.type}")
                stream = data.pop(name).astype(np.uint8, copy=False)
                if stream.shape[1:] != (256,):
                    raise ValueError(f"{name} must have width 256")
                streams.append(stream)
            data["tactile"] = np.concatenate(streams, axis=-1)
            if data["tactile"].shape[1:] != (TACTILE_DIM,):
                raise ValueError(f"concatenated tactile must have width {TACTILE_DIM}")
        self._parquet_cache[episode_index] = data
        while len(self._parquet_cache) > self._parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return data

    def _decode_frames(self, path: Path, indices: Iterable[int]) -> torch.Tensor:
        """Decode nearby frame indices with one seek; output is float [T,C,H,W]."""
        import av

        requested = list(indices)
        if not requested:
            raise ValueError("at least one video frame must be requested")
        target_seconds = [index / self.fps for index in requested]
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            seek_seconds = max(0.0, target_seconds[0] - 1.0)
            container.seek(int(seek_seconds / float(stream.time_base)), stream=stream)
            target_position = 0
            last = None
            tolerance = 0.5 / self.fps
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                seconds = float(frame.pts * stream.time_base)
                last = frame
                while (
                    target_position < len(target_seconds)
                    and seconds + tolerance >= target_seconds[target_position]
                ):
                    frames.append(frame.to_ndarray(format="rgb24"))
                    target_position += 1
                if target_position == len(target_seconds):
                    break
            if len(frames) != len(requested):
                if last is None:
                    raise RuntimeError(f"could not decode frames from {path}")
                last_image = last.to_ndarray(format="rgb24")
                frames.extend([last_image] * (len(requested) - len(frames)))
        tensor = (
            torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
        )
        return F.interpolate(
            tensor,
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_position, frame_index = self._locate(index)
        episode_index = int(self.episodes[episode_position]["episode_index"])
        data = self._load_episode(episode_index)
        future = np.arange(frame_index + 1, frame_index + self.config.dream_horizon + 1)
        action_slice = slice(frame_index, frame_index + ACTION_HORIZON)

        states = assemble_state_46(
            data["observation.state"],
            data["observation.projected_gravity"],
            self.state_spans,
        )
        actions = assemble_action_78(
            data["action.motion_token"],
            data["teleop.left_hand_joints"],
            data["teleop.right_hand_joints"],
        )
        image_indices = [frame_index]
        if self.config.mode.dreams_state_and_vision:
            image_indices.extend(future.tolist())
        images = []
        for video_key in (
            "observation.images.ego_view_left",
            "observation.images.ego_view_right",
        ):
            path = self._format_path(
                self.info["video_path"], episode_index, video_key=video_key
            )
            images.append(self._decode_frames(path, image_indices))
        stereo = torch.stack(images, dim=1)

        task_index = int(data["task_index"][frame_index])
        sample = {
            "state": torch.from_numpy(states[frame_index].copy()),
            "images": stereo[0],
            "prompt_tokens": torch.from_numpy(
                tokenize_prompt(self.tasks[task_index], self.config.max_prompt_bytes)
            ),
            "actions": torch.from_numpy(actions[action_slice].copy()),
        }
        if self.config.mode.uses_tactile:
            sample["tactile"] = torch.from_numpy(
                data["tactile"][frame_index].copy()
            )
            sample["future_tactile"] = torch.from_numpy(
                data["tactile"][future].copy()
            )
        if self.config.mode.dreams_state_and_vision:
            sample["future_state"] = torch.from_numpy(states[future].copy())
            sample["future_images"] = stereo[1:]
        return sample
