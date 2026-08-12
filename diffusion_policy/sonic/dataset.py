from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import numpy as np
import torch
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
VIDEO_KEYS = (
    "observation.images.ego_view_left",
    "observation.images.ego_view_right",
)
VIDEO_CACHE_VERSION = 1
LOWDIM_CACHE_VERSION = 1


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _video_cache_root(
    dataset_path: str | Path,
    config: SonicConfig,
    cache_dir: str | Path | None = None,
) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return (
        Path(dataset_path)
        / ".cache"
        / "diffusion_policy"
        / f"rgb_{config.image_height}x{config.image_width}_v{VIDEO_CACHE_VERSION}"
    )


def _lowdim_cache_root(dataset_path: str | Path) -> Path:
    return (
        Path(dataset_path)
        / ".cache"
        / "diffusion_policy"
        / f"lowdim_v{LOWDIM_CACHE_VERSION}"
    )


def _valid_lowdim_cache(cache_root: Path, total_frames: int) -> bool:
    expected = {
        "state.npy": (np.float32, (total_frames, STATE_DIM)),
        "action.npy": (np.float32, (total_frames, ACTION_DIM)),
        "tactile.npy": (np.uint8, (total_frames, TACTILE_DIM)),
        "task_index.npy": (np.int64, (total_frames,)),
    }
    try:
        for filename, (dtype, shape) in expected.items():
            values = np.load(cache_root / filename, mmap_mode="r", allow_pickle=False)
            if values.dtype != dtype or values.shape != shape:
                return False
        return True
    except (OSError, ValueError):
        return False


def build_lowdim_cache(dataset_path: str | Path, *, progress: bool = True) -> Path:
    """Convert per-episode parquet columns once into compact global memory maps."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(dataset_path)
    info = _read_json(root / "meta" / "info.json")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    total_frames = sum(int(episode["length"]) for episode in episodes)
    cache_root = _lowdim_cache_root(root)
    if _valid_lowdim_cache(cache_root, total_frames):
        return cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_suffix = f".{os.getpid()}.tmp.npy"
    arrays = {
        "state": np.lib.format.open_memmap(
            cache_root / f"state{temporary_suffix}",
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, STATE_DIM),
        ),
        "action": np.lib.format.open_memmap(
            cache_root / f"action{temporary_suffix}",
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, ACTION_DIM),
        ),
        "tactile": np.lib.format.open_memmap(
            cache_root / f"tactile{temporary_suffix}",
            mode="w+",
            dtype=np.uint8,
            shape=(total_frames, TACTILE_DIM),
        ),
        "task_index": np.lib.format.open_memmap(
            cache_root / f"task_index{temporary_suffix}",
            mode="w+",
            dtype=np.int64,
            shape=(total_frames,),
        ),
    }
    state_spans = load_state_spans(root)
    chunk_size = int(info.get("chunks_size", 1000))
    offset = 0
    try:
        for position, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            path = root / info["data_path"].format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
            table = pq.read_table(path, columns=list(PARQUET_COLUMNS))
            if len(table) != length:
                raise ValueError(f"{path} has {len(table)} rows, expected {length}")
            target = slice(offset, offset + length)
            arrays["state"][target] = assemble_state_46(
                np.asarray(table["observation.state"].to_pylist()),
                np.asarray(table["observation.projected_gravity"].to_pylist()),
                state_spans,
            )
            arrays["action"][target] = assemble_action_78(
                np.asarray(table["action.motion_token"].to_pylist()),
                np.asarray(table["teleop.left_hand_joints"].to_pylist()),
                np.asarray(table["teleop.right_hand_joints"].to_pylist()),
            )
            streams = []
            for name in TACTILE_COLUMNS:
                column = table[name]
                if not pa.types.is_list(column.type) or not pa.types.is_uint8(column.type.value_type):
                    raise ValueError(f"{name} must be list<uint8>, got {column.type}")
                stream = np.asarray(column.to_pylist(), dtype=np.uint8)
                if stream.shape != (length, 256):
                    raise ValueError(f"{name} must have shape ({length}, 256)")
                streams.append(stream)
            arrays["tactile"][target] = np.concatenate(streams, axis=-1)
            arrays["task_index"][target] = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
            offset += length
            if progress and (position == len(episodes) or position % 8 == 0):
                print(f"Low-dimensional cache: {position}/{len(episodes)}", flush=True)
        for values in arrays.values():
            values.flush()
        del arrays
        for name in ("state", "action", "tactile", "task_index"):
            os.replace(
                cache_root / f"{name}{temporary_suffix}",
                cache_root / f"{name}.npy",
            )
    except BaseException:
        del arrays
        for name in ("state", "action", "tactile", "task_index"):
            (cache_root / f"{name}{temporary_suffix}").unlink(missing_ok=True)
        raise
    return cache_root


def _cached_video_path(cache_root: Path, video_key: str, episode_index: int) -> Path:
    return cache_root / video_key / f"episode_{episode_index:06d}.npy"


def _valid_cached_video(path: Path, length: int, height: int, width: int) -> bool:
    if not path.is_file():
        return False
    try:
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        return frames.dtype == np.uint8 and frames.shape == (length, height, width, 3)
    except (OSError, ValueError):
        return False


def _decode_video_to_cache(
    source: Path,
    destination: Path,
    *,
    length: int,
    height: int,
    width: int,
) -> None:
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp.npy")
    frames = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(length, height, width, 3),
    )
    decoded = 0
    last_frame = None
    try:
        with av.open(str(source)) as container:
            for frame in container.decode(video=0):
                if decoded >= length:
                    break
                last_frame = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                frames[decoded] = last_frame
                decoded += 1
        if decoded == 0:
            raise RuntimeError(f"could not decode frames from {source}")
        if decoded < length:
            frames[decoded:length] = last_frame
        frames.flush()
        del frames
        os.replace(temporary, destination)
    except BaseException:
        del frames
        temporary.unlink(missing_ok=True)
        raise


def build_video_cache(
    dataset_path: str | Path,
    config: SonicConfig,
    cache_dir: str | Path | None = None,
    *,
    progress: bool = True,
) -> Path:
    """Sequentially decode each stereo episode once into mmap-friendly uint8 arrays."""
    root = Path(dataset_path)
    info = _read_json(root / "meta" / "info.json")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    chunk_size = int(info.get("chunks_size", 1000))
    cache_root = _video_cache_root(root, config, cache_dir)
    total = len(episodes) * len(VIDEO_KEYS)
    completed = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        for video_key in VIDEO_KEYS:
            destination = _cached_video_path(cache_root, video_key, episode_index)
            if not _valid_cached_video(
                destination, length, config.image_height, config.image_width
            ):
                source = root / info["video_path"].format(
                    episode_chunk=episode_index // chunk_size,
                    episode_index=episode_index,
                    video_key=video_key,
                )
                _decode_video_to_cache(
                    source,
                    destination,
                    length=length,
                    height=config.image_height,
                    width=config.image_width,
                )
            completed += 1
            if progress and (completed == total or completed % 8 == 0):
                print(f"Video cache: {completed}/{total}", flush=True)
    return cache_root


def build_training_cache(
    dataset_path: str | Path,
    config: SonicConfig,
    cache_dir: str | Path | None = None,
    *,
    progress: bool = True,
) -> tuple[Path, Path]:
    lowdim_root = build_lowdim_cache(dataset_path, progress=progress)
    video_root = build_video_cache(
        dataset_path, config, cache_dir, progress=progress
    )
    return lowdim_root, video_root


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
        video_cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(dataset_path)
        self.config = config
        self.info = _read_json(self.root / "meta" / "info.json")
        self.fps = float(self.info["fps"])
        self.chunk_size = int(self.info.get("chunks_size", 1000))
        tasks = _read_jsonl(self.root / "meta" / "tasks.jsonl")
        self.tasks = {int(item["task_index"]): item["task"] for item in tasks}
        self.episodes = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.episode_lengths = {
            int(episode["episode_index"]): int(episode["length"])
            for episode in self.episodes
        }
        self.valid_lengths = []
        self.episode_offsets = {}
        offset = 0
        for episode in self.episodes:
            episode_index = int(episode["episode_index"])
            self.episode_offsets[episode_index] = offset
            offset += int(episode["length"])
            valid = int(episode["length"]) - config.action_horizon + 1
            self.valid_lengths.append(max(0, valid))
        self.cumulative_lengths = np.cumsum(self.valid_lengths).tolist()
        del parquet_cache_size
        lowdim_root = _lowdim_cache_root(self.root)
        if not _valid_lowdim_cache(lowdim_root, offset):
            raise FileNotFoundError(
                f"low-dimensional cache is missing or invalid: {lowdim_root}; run training once to build it"
            )
        self.state = np.load(lowdim_root / "state.npy", mmap_mode="r", allow_pickle=False)
        self.action = np.load(lowdim_root / "action.npy", mmap_mode="r", allow_pickle=False)
        self.tactile = np.load(lowdim_root / "tactile.npy", mmap_mode="r", allow_pickle=False)
        self.task_index = np.load(lowdim_root / "task_index.npy", mmap_mode="r", allow_pickle=False)
        self.video_cache_root = _video_cache_root(self.root, config, video_cache_dir)
        self._video_cache: dict[tuple[str, int], np.ndarray] = {}
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

    def _cached_frames(
        self, video_key: str, episode_index: int, indices: list[int]
    ) -> torch.Tensor:
        cache_key = (video_key, episode_index)
        frames = self._video_cache.get(cache_key)
        if frames is None:
            path = _cached_video_path(self.video_cache_root, video_key, episode_index)
            expected_length = self.episode_lengths[episode_index]
            if not _valid_cached_video(
                path,
                expected_length,
                self.config.image_height,
                self.config.image_width,
            ):
                raise FileNotFoundError(
                    f"video cache is missing or invalid: {path}; run training once to build it"
                )
            frames = np.load(path, mmap_mode="r", allow_pickle=False)
            self._video_cache[cache_key] = frames
        # Advanced indexing returns a writable compact copy, avoiding undefined behavior in
        # torch.from_numpy for read-only memory maps.
        array = np.asarray(frames[indices]).copy()
        return torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_position, frame_index = self._locate(index)
        episode_index = int(self.episodes[episode_position]["episode_index"])
        absolute_index = self.episode_offsets[episode_index] + frame_index
        future = np.arange(frame_index + 1, frame_index + self.config.dream_horizon + 1)
        absolute_future = absolute_index + np.arange(1, self.config.dream_horizon + 1)
        action_slice = slice(absolute_index, absolute_index + ACTION_HORIZON)
        image_indices = [frame_index]
        if self.config.mode.dreams_state_and_vision:
            image_indices.extend(future.tolist())
        images = []
        for video_key in VIDEO_KEYS:
            images.append(self._cached_frames(video_key, episode_index, image_indices))
        stereo = torch.stack(images, dim=1)

        task_index = int(self.task_index[absolute_index])
        sample = {
            "state": torch.from_numpy(self.state[absolute_index].copy()),
            "images": stereo[0],
            "prompt_tokens": torch.from_numpy(
                tokenize_prompt(self.tasks[task_index], self.config.max_prompt_bytes)
            ),
            "actions": torch.from_numpy(self.action[action_slice].copy()),
        }
        if self.config.mode.uses_tactile:
            sample["tactile"] = torch.from_numpy(self.tactile[absolute_index].copy())
            sample["future_tactile"] = torch.from_numpy(self.tactile[absolute_future].copy())
        if self.config.mode.dreams_state_and_vision:
            sample["future_state"] = torch.from_numpy(self.state[absolute_future].copy())
            sample["future_images"] = stereo[1:]
        return sample
