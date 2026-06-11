import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Dataset

from autoshotv2.model.linear import Linear_
from autoshotv2.utils import (
    evaluate_scenes,
    get_batches,
    get_frames,
    mAP_f1_p_fix_r,
    predictions_to_scenes,
    scenes2zero_one_representation,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

DEFAULT_META = "./shot_clipshots_trainval.pickle"
DEFAULT_BASE_CKPT = "./artifacts/models/ckpt_0_200_0.pth"
DEFAULT_OUT_CKPT = "./artifacts/models/training/ckpt_phase2_shot_clipshots_best.pth"
DEFAULT_SAMPLE_CACHE = "./shot_clipshots_phase2_sample_cache.pkl"
DEFAULT_RESULTS = "./phase2_shot_clipshots_results.pkl"
DEFAULT_EVAL_CACHE_DIR = "./eval_cache_shot_clipshots"
DEFAULT_RESUME_STATE = "./artifacts/models/training/phase2_shot_clipshots_resume.pt"
DEFAULT_CHECKPOINT_DIR = "./artifacts/models/training/phase2_shot_clipshots_checkpoints"
SAMPLE_CACHE_SCHEMA_VERSION = 2


device = "cuda" if torch.cuda.is_available() else "cpu"


class TimeBudgetExpired(Exception):
    pass


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        a_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (a_t * (1.0 - p_t) ** self.gamma * ce).mean()


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int = 4864, hidden_dim: int = 1024, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = Linear_(in_features, hidden_dim, bias=True, act="ReLU")
        # dropout_rate is the probability of zeroing an element (PyTorch convention),
        # not a keep-rate. Default 0.5 keeps the original behaviour unchanged.
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cls_layer1 = Linear_(hidden_dim, 1, bias=True, act="Identity")
        self.cls_layer2 = Linear_(hidden_dim, 1, bias=True, act="Identity")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = self.dropout(x)
        return self.cls_layer1(x), self.cls_layer2(x)


class SampleFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, one_hot: torch.Tensor, boundary: torch.Tensor):
        self.features = features
        self.one_hot = one_hot
        self.boundary = boundary

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx].float(),
            self.one_hot[idx].float().unsqueeze(0),
            self.boundary[idx].float().unsqueeze(0),
        )


class ManualAdam:
    """Small Adam implementation to avoid torch.optim importing torch._dynamo/onnx."""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        for param, exp_avg, exp_avg_sq in zip(self.params, self.exp_avg, self.exp_avg_sq):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay != 0.0:
                grad = grad.add(param, alpha=self.weight_decay)
            exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(self.eps)
            step_size = self.lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "step_count": self.step_count,
            "exp_avg": [t.detach().cpu() for t in self.exp_avg],
            "exp_avg_sq": [t.detach().cpu() for t in self.exp_avg_sq],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr = float(state["lr"])
        self.weight_decay = float(state["weight_decay"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.step_count = int(state["step_count"])
        for dst, src in zip(self.exp_avg, state["exp_avg"]):
            dst.copy_(src.to(dst.device))
        for dst, src in zip(self.exp_avg_sq, state["exp_avg_sq"]):
            dst.copy_(src.to(dst.device))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_keys(keys: list[str]) -> str:
    h = hashlib.sha256()
    for key in sorted(keys):
        h.update(key.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def select_training_keys(
    train_keys: list[str],
    seed: int,
    max_train_videos: int,
) -> list[str]:
    """Return the exact, deterministic video order used to build the sample cache."""
    selected = list(train_keys)
    random.Random(seed).shuffle(selected)
    if max_train_videos > 0:
        selected = selected[:max_train_videos]
    return selected


def build_sample_cache_config(
    meta_path: str,
    selected_keys: list[str],
    base_ckpt_hash: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a strict cache identity from every input that changes sampled data."""
    return {
        "schema_version": SAMPLE_CACHE_SCHEMA_VERSION,
        "meta_path": os.path.abspath(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "selected_keys_hash": hash_keys(selected_keys),
        "selected_keys_count": len(selected_keys),
        "base_ckpt_hash": base_ckpt_hash,
        "max_train_videos": args.max_train_videos,
        "max_samples_per_video": args.max_samples_per_video,
        "max_total_samples": args.max_total_samples,
        "neg_per_pos": args.neg_per_pos,
        "min_neg_per_video": args.min_neg_per_video,
        "boundary_window": args.boundary_window,
        "max_cache_video_frames": args.max_cache_video_frames,
        "max_cache_video_seconds": args.max_cache_video_seconds,
        "data_seed": args.data_seed,
    }


def write_training_data_manifest(
    path: str,
    meta_path: str,
    base_ckpt_path: str,
    entries: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    data_seed: int,
) -> dict[str, Any]:
    selected_keys = list(stats["selected_keys"])
    completed_keys = set(stats["completed_keys"])
    skipped_by_key = {str(key): str(reason) for key, reason in stats["skipped"]}
    rows = []
    dataset_counts: dict[str, int] = {}
    for key in selected_keys:
        entry = entries[key]
        dataset = str(entry.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        status = (
            "completed"
            if key in completed_keys
            else "skipped"
            if key in skipped_by_key
            else "not_processed"
        )
        rows.append(
            {
                "key": key,
                "dataset": dataset,
                "source_split": entry.get("source_split"),
                "source_name": entry.get("source_name"),
                "status": status,
                "samples": int(stats["sample_counts"].get(key, 0)),
                "skip_reason": skipped_by_key.get(key),
            }
        )
    payload = {
        "schema_version": 1,
        "data_seed": data_seed,
        "metadata": {
            "path": os.path.abspath(meta_path),
            "sha256": sha256_file(meta_path),
        },
        "base_checkpoint": {
            "path": os.path.abspath(base_ckpt_path),
            "sha256": sha256_file(base_ckpt_path),
        },
        "selected_keys_hash": hash_keys(selected_keys),
        "selected_videos": len(selected_keys),
        "completed_videos": len(completed_keys),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "sampling": stats["cache_config"],
        "videos": rows,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def hash_head_state(head: ClassificationHead) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_boundary_labels(one_hot: np.ndarray, window: int) -> np.ndarray:
    labels = one_hot.copy().astype(np.float32)
    for idx in np.flatnonzero(one_hot > 0):
        start = max(0, idx - window)
        end = min(len(labels), idx + window + 1)
        labels[start:end] = 1.0
    return labels


def transitions_to_scenes(transitions: np.ndarray, n_frames: int) -> np.ndarray:
    transitions = np.asarray(transitions, dtype=np.int32)
    if n_frames <= 0:
        return np.asarray([[0, 0]], dtype=np.int32)
    if transitions.size == 0:
        return np.asarray([[0, n_frames - 1]], dtype=np.int32)

    transitions = transitions.reshape(-1, 2)
    transitions = transitions[np.argsort(transitions[:, 0])]
    transitions = np.clip(transitions, 0, n_frames - 1)
    transitions = transitions[transitions[:, 0] <= transitions[:, 1]]
    if len(transitions) == 0:
        return np.asarray([[0, n_frames - 1]], dtype=np.int32)

    scenes = [[0, int(transitions[0, 0])]]
    for i in range(1, len(transitions)):
        scenes.append([int(transitions[i - 1, 1]), int(transitions[i, 0])])
    scenes.append([int(transitions[-1, 1]), n_frames - 1])

    arr = np.asarray(scenes, dtype=np.int32)
    arr = np.clip(arr, 0, n_frames - 1)
    arr = arr[arr[:, 0] <= arr[:, 1]]
    if len(arr) == 0:
        arr = np.asarray([[0, n_frames - 1]], dtype=np.int32)
    return arr


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    required = {"entries", "train_keys", "val_keys", "shot_test_entries"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Metadata file is missing fields: {sorted(missing)}")
    return payload


def load_supernet(ckpt_path: str):
    from autoshotv2.model.supernet import TransNetV2Supernet

    model = TransNetV2Supernet().eval().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = model.state_dict()
    pretrained = {k: v for k, v in ckpt["net"].items() if k in sd}
    sd.update(pretrained)
    model.load_state_dict(sd)
    print(f"Loaded {len(pretrained)}/{len(sd)} tensors from {ckpt_path}")
    return model


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    return float(value)


def probe_video_info(video_path: str) -> dict[str, float | int | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {"probe_error": str(exc), "duration": None, "fps": None, "frames": None, "size": None}

    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    duration_raw = stream.get("duration") or fmt.get("duration")
    duration = None if duration_raw in {None, "N/A"} else float(duration_raw)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))

    frames_raw = stream.get("nb_frames")
    frames = int(frames_raw) if frames_raw not in {None, "N/A"} else None
    if frames is None and duration is not None and fps is not None:
        frames = int(duration * fps)

    size_raw = fmt.get("size")
    size = int(size_raw) if size_raw not in {None, "N/A"} else None
    return {"probe_error": None, "duration": duration, "fps": fps, "frames": frames, "size": size}


def check_sample_cache_video_budget(video_path: str, args: argparse.Namespace) -> None:
    info = probe_video_info(video_path)
    frames = info.get("frames")
    duration = info.get("duration")
    fps = info.get("fps")

    if frames is not None and args.max_cache_video_frames > 0 and frames > args.max_cache_video_frames:
        raise RuntimeError(
            "video too long for sample cache: "
            f"frames={frames} fps={fps} duration={duration} "
            f"limit_frames={args.max_cache_video_frames} path={video_path}"
        )

    if duration is not None and args.max_cache_video_seconds > 0 and duration > args.max_cache_video_seconds:
        raise RuntimeError(
            "video too long for sample cache: "
            f"duration={duration:.1f}s frames={frames} fps={fps} "
            f"limit_seconds={args.max_cache_video_seconds} path={video_path}"
        )

    if info.get("probe_error"):
        raise RuntimeError(f"ffprobe failed before sample-cache decode: {info['probe_error']} path={video_path}")


def extract_backbone_features(backbone, video_path: str, captured: dict[str, torch.Tensor]) -> torch.Tensor:
    frames = get_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError(f"No decoded frames: {video_path}")

    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in get_batches(frames):
            t = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
            captured.clear()
            backbone(t)
            if "feat" not in captured:
                raise RuntimeError("fc1_0 hook did not capture features")
            chunks.append(captured["feat"][0, 25:75, :].cpu())
    return torch.cat(chunks, 0)[: len(frames)]


def select_sample_indices(
    one_hot: np.ndarray,
    boundary: np.ndarray,
    max_samples_per_video: int,
    neg_per_pos: int,
    min_neg_per_video: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pos_idx = np.unique(np.concatenate([np.flatnonzero(one_hot > 0), np.flatnonzero(boundary > 0)]))
    neg_idx = np.flatnonzero(boundary == 0)

    if max_samples_per_video <= 0:
        max_samples_per_video = len(one_hot)

    if len(pos_idx) > 0:
        pos_cap = max(1, max_samples_per_video // max(neg_per_pos + 1, 1))
        if len(pos_idx) > pos_cap:
            pos_idx = rng.choice(pos_idx, size=pos_cap, replace=False)
        n_neg = min(len(neg_idx), max_samples_per_video - len(pos_idx), max(min_neg_per_video, len(pos_idx) * neg_per_pos))
    else:
        n_neg = min(len(neg_idx), max_samples_per_video, min_neg_per_video)

    if n_neg > 0:
        neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
        selected = np.concatenate([pos_idx, neg_idx])
    else:
        selected = pos_idx

    if len(selected) == 0:
        return selected.astype(np.int64)
    rng.shuffle(selected)
    return selected.astype(np.int64)


def deadline_expired(args: argparse.Namespace) -> bool:
    deadline = getattr(args, "_deadline", None)
    return deadline is not None and time.monotonic() >= deadline


def sample_cache_partial_paths(cache_path: str) -> tuple[str, str]:
    return cache_path + ".parts", cache_path + ".partial.pkl"


def build_or_load_sample_cache(
    cache_path: str,
    meta_path: str,
    entries: dict[str, dict[str, Any]],
    train_keys: list[str],
    backbone,
    base_ckpt_hash: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    work_keys = select_training_keys(train_keys, args.data_seed, args.max_train_videos)
    cache_config = build_sample_cache_config(
        meta_path,
        work_keys,
        base_ckpt_hash,
        args,
    )
    parts_dir, partial_manifest_path = sample_cache_partial_paths(cache_path)

    if os.path.exists(cache_path) and not args.rebuild_sample_cache:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("config") == cache_config:
            print(f"Loading sample cache: {cache_path}")
            return cached["features"], cached["one_hot"], cached["boundary"], cached["stats"]
        print("Sample cache identity changed; rebuilding.")

    rng = np.random.default_rng(args.data_seed)

    done_keys: set[str] = set()
    part_files: list[str] = []
    skipped: list[tuple[str, str]] = []
    sample_counts: dict[str, int] = {}
    total_samples = 0

    if os.path.exists(partial_manifest_path) and not args.rebuild_sample_cache:
        with open(partial_manifest_path, "rb") as f:
            manifest = pickle.load(f)
        if manifest.get("config") == cache_config:
            done_keys = set(manifest.get("done_keys", []))
            part_files = list(manifest.get("part_files", []))
            skipped = list(manifest.get("skipped", []))
            sample_counts = {
                str(key): int(value)
                for key, value in manifest.get("sample_counts", {}).items()
            }
            total_samples = int(manifest.get("samples", 0))
            print(f"Resuming partial sample cache: done_videos={len(done_keys)} samples={total_samples}")
        else:
            print("Partial sample cache exists but config changed; starting over.")

    os.makedirs(parts_dir, exist_ok=True)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inp, _out):
        captured["feat"] = inp[0].detach().cpu()

    handle = backbone.fc1_0.register_forward_hook(hook)
    chunk_features: list[torch.Tensor] = []
    chunk_one_hot: list[torch.Tensor] = []
    chunk_boundary: list[torch.Tensor] = []
    chunk_keys: list[str] = []

    def write_manifest() -> None:
        with open(partial_manifest_path, "wb") as f:
            pickle.dump(
                {
                    "config": cache_config,
                    "done_keys": sorted(done_keys),
                    "part_files": part_files,
                    "skipped": skipped,
                    "sample_counts": sample_counts,
                    "samples": total_samples,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def flush_chunk() -> None:
        nonlocal chunk_features, chunk_one_hot, chunk_boundary, chunk_keys
        if not chunk_features:
            write_manifest()
            return
        part_path = os.path.join(parts_dir, f"part_{len(part_files):05d}.pt")
        torch.save(
            {
                "features": torch.cat(chunk_features, 0),
                "one_hot": torch.cat(chunk_one_hot, 0),
                "boundary": torch.cat(chunk_boundary, 0),
                "keys": chunk_keys,
            },
            part_path,
        )
        part_files.append(part_path)
        chunk_features = []
        chunk_one_hot = []
        chunk_boundary = []
        chunk_keys = []
        write_manifest()
        print(f"  partial sample cache saved: {part_path}", flush=True)

    try:
        for i, key in enumerate(work_keys, 1):
            if key in done_keys:
                continue
            if args.max_total_samples > 0 and total_samples >= args.max_total_samples:
                break

            entry = entries[key]
            try:
                print(f"  sample cache processing [{i}/{len(work_keys)}] key={key}", flush=True)
                check_sample_cache_video_budget(entry["video_path"], args)
                feats = extract_backbone_features(backbone, entry["video_path"], captured)
                n_frames = int(feats.shape[0])
                scenes = transitions_to_scenes(entry["transitions"], n_frames)
                one_hot_np, _ = scenes2zero_one_representation(scenes, n_frames)
                boundary_np = make_boundary_labels(one_hot_np, args.boundary_window)
                idx = select_sample_indices(
                    one_hot_np,
                    boundary_np,
                    args.max_samples_per_video,
                    args.neg_per_pos,
                    args.min_neg_per_video,
                    rng,
                )
                if len(idx) == 0:
                    skipped.append((key, "no sampled indices"))
                    continue
                if args.max_total_samples > 0 and total_samples + len(idx) > args.max_total_samples:
                    keep = args.max_total_samples - total_samples
                    idx = rng.choice(idx, size=keep, replace=False)

                chunk_features.append(feats[idx].half())
                chunk_one_hot.append(torch.from_numpy(one_hot_np[idx].astype(np.float32)))
                chunk_boundary.append(torch.from_numpy(boundary_np[idx].astype(np.float32)))
                chunk_keys.append(key)
                total_samples += len(idx)
                sample_counts[key] = len(idx)
                done_keys.add(key)
                if i == 1 or i % 25 == 0:
                    pct = 100.0 * i / max(len(work_keys), 1)
                    print(f"  sample cache {pct:6.2f}% [{i}/{len(work_keys)}] samples={total_samples} key={key}", flush=True)
                if args.save_every_videos > 0 and len(chunk_keys) >= args.save_every_videos:
                    flush_chunk()
                if deadline_expired(args):
                    flush_chunk()
                    raise TimeBudgetExpired("Time budget reached while building sample cache.")
            except Exception as exc:
                skipped.append((key, str(exc)))
                print(f"  [skip] {key}: {exc}")
                done_keys.add(key)
                if deadline_expired(args):
                    flush_chunk()
                    raise TimeBudgetExpired("Time budget reached while building sample cache.")
    finally:
        handle.remove()

    flush_chunk()

    if not part_files:
        raise RuntimeError("No training samples were extracted.")

    loaded_parts = [torch.load(path, map_location="cpu", weights_only=False) for path in part_files]
    x = torch.cat([part["features"] for part in loaded_parts], 0)
    y1 = torch.cat([part["one_hot"] for part in loaded_parts], 0)
    y2 = torch.cat([part["boundary"] for part in loaded_parts], 0)
    stats = {
        "videos_seen": len(work_keys),
        "videos_completed": len(done_keys),
        "samples": int(x.shape[0]),
        "one_hot_positive_rate": float(y1.mean().item()),
        "boundary_positive_rate": float(y2.mean().item()),
        "skipped": skipped,
        "selected_keys": work_keys,
        "selected_keys_hash": hash_keys(work_keys),
        "completed_keys": sorted(done_keys),
        "sample_counts": dict(sorted(sample_counts.items())),
        "cache_config": cache_config,
        "partial_parts": part_files,
    }

    with open(cache_path, "wb") as f:
        pickle.dump({"config": cache_config, "features": x, "one_hot": y1, "boundary": y2, "stats": stats}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Sample cache saved -> {cache_path}")
    return x, y1, y2, stats


def train_head(
    head: ClassificationHead,
    dataset: SampleFeatureDataset,
    args: argparse.Namespace,
) -> tuple[ClassificationHead, list[float]]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    optimizer = ManualAdam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    ema_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    train_config = {
        "in_features": int(dataset.features.shape[1]),
        "n_samples": int(len(dataset)),
        "batch_size": args.batch_size,
        "loss": args.loss,
        "lr": args.lr,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "weight_decay": args.weight_decay,
        "manyhot_weight": args.manyhot_weight,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "training_seed": args.seed,
    }

    if os.path.exists(args.resume_state) and not args.no_resume:
        state = torch.load(args.resume_state, map_location=device, weights_only=False)
        if state.get("train_config") != train_config and not args.ignore_resume_config:
            raise RuntimeError(
                f"Resume config mismatch in {args.resume_state}. "
                "Use --no-resume or --ignore-resume-config if this is intentional."
            )
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        losses = list(state.get("losses", []))
        start_epoch = int(state.get("epoch", 0)) + 1
        if args.use_ema and state.get("ema_state") is not None:
            ema_state = {k: v.to(device) for k, v in state["ema_state"].items()}
        print(f"Resuming training from epoch {start_epoch}/{args.epochs}")

    if args.use_ema and ema_state is None:
        ema_state = {k: v.detach().clone().to(device) for k, v in head.state_dict().items()}

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_training_state(epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "head": head.state_dict(),
            "ema_state": {k: v.detach().cpu() for k, v in ema_state.items()} if ema_state is not None else None,
            "optimizer": optimizer.state_dict(),
            "losses": losses,
            "train_config": train_config,
            "args": vars(args),
        }
        torch.save(payload, args.resume_state)
        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0 or epoch == args.epochs):
            torch.save(payload, os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt"))

    if start_epoch > args.epochs:
        print(f"Training already complete at epoch {start_epoch - 1}.")
        if ema_state is not None:
            head.load_state_dict({k: v.detach().cpu() for k, v in ema_state.items()})
        return head, losses

    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        total_loss = 0.0
        n_seen = 0
        num_batches = max(len(loader), 1)
        for batch_idx, (feats, one_hot, boundary) in enumerate(loader, 1):
            feats = feats.to(device)
            one_hot = one_hot.to(device)
            boundary = boundary.to(device)
            logits1, logits2 = head(feats)
            loss = criterion(logits1, one_hot) + args.manyhot_weight * criterion(logits2, boundary)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_state is not None:
                with torch.no_grad():
                    for name, value in head.state_dict().items():
                        ema_state[name].mul_(args.ema_decay).add_(value.detach(), alpha=1.0 - args.ema_decay)

            total_loss += float(loss.item()) * len(feats)
            n_seen += len(feats)
            if batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == num_batches:
                overall_pct = 100.0 * ((epoch - 1) + batch_idx / num_batches) / max(args.epochs, 1)
                running_loss = total_loss / max(n_seen, 1)
                print(
                    f"train {overall_pct:6.2f}% epoch {epoch:03d}/{args.epochs} "
                    f"batch {batch_idx:05d}/{num_batches:05d} loss={running_loss:.6f}",
                    flush=True,
                )

        avg_loss = total_loss / max(n_seen, 1)
        losses.append(avg_loss)
        print(f"epoch {epoch:03d}/{args.epochs} done loss={avg_loss:.6f}", flush=True)
        save_training_state(epoch)
        if deadline_expired(args):
            raise TimeBudgetExpired("Time budget reached after saving epoch checkpoint.")
    if ema_state is not None:
        head.load_state_dict({k: v.detach().cpu() for k, v in ema_state.items()})
    return head, losses


def merge_head_into_state(base_state: dict[str, torch.Tensor], head: ClassificationHead) -> dict[str, torch.Tensor]:
    full_sd = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    for k, v in head.fc1.state_dict().items():
        full_sd[f"fc1_0.{k}"] = v.detach().cpu()
    for k, v in head.cls_layer1.state_dict().items():
        full_sd[f"cls_layer1.{k}"] = v.detach().cpu()
    for k, v in head.cls_layer2.state_dict().items():
        full_sd[f"cls_layer2.{k}"] = v.detach().cpu()
    return full_sd


def load_model_from_state(full_sd: dict[str, torch.Tensor]):
    from autoshotv2.model.supernet import TransNetV2Supernet

    model = TransNetV2Supernet().eval().to(device)
    model.load_state_dict(full_sd)
    return model


def predict_video_logits(model, video_path: str) -> np.ndarray:
    frames = get_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError(f"No decoded frames: {video_path}")
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in get_batches(frames):
            t = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
            out = model(t)
            if isinstance(out, tuple):
                out = out[0]
            chunks.append(out[0].detach().cpu().numpy()[25:75])
    logits = np.concatenate(chunks, 0)[: len(frames)]
    if logits.ndim == 1:
        logits = logits[:, np.newaxis]
    return logits.astype(np.float32)


def load_or_run_logits(
    model,
    entries: dict[str, dict[str, Any]],
    keys: list[str],
    cache_path: str,
    cache_config: dict[str, Any],
    no_cache: bool,
) -> dict[str, np.ndarray]:
    if os.path.exists(cache_path) and not no_cache:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("config") == cache_config:
            print(f"Loading logits cache: {cache_path}")
            return cached["logits"]
        print(f"Logits cache config changed, rebuilding: {cache_path}")

    logits: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for i, key in enumerate(keys, 1):
        entry = entries[key]
        if not os.path.exists(entry["video_path"]):
            missing.append(key)
            continue
        pct = 100.0 * i / max(len(keys), 1)
        print(f"  inference {pct:6.2f}% [{i}/{len(keys)}] {key}", flush=True)
        logits[key] = predict_video_logits(model, entry["video_path"])

    if missing:
        print(f"WARNING: {len(missing)} videos missing.")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"config": cache_config, "logits": logits}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Logits cache saved -> {cache_path}")
    return logits


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logits_to_pred_dict(logits: dict[str, np.ndarray], temperature: float, sigma: float) -> dict[str, np.ndarray]:
    pred: dict[str, np.ndarray] = {}
    for key, arr in logits.items():
        probs = sigmoid_np(arr / temperature).squeeze()
        if sigma > 0:
            probs = gaussian_filter1d(probs, sigma=sigma)
        pred[key] = probs[:, np.newaxis].astype(np.float32)
    return pred


def gt_for_logits(entries: dict[str, dict[str, Any]], logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: transitions_to_scenes(entries[key]["transitions"], len(value))
        for key, value in logits.items()
        if key in entries
    }


def evaluate_best(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict[str, float]:
    common_pred = {k: v for k, v in pred.items() if k in gt}
    common_gt = {k: gt[k] for k in common_pred}
    _, f1, precision, recall, threshold, _ = mAP_f1_p_fix_r(common_pred, common_gt, fixed_r=-1)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "n_videos": len(common_pred),
    }


def evaluate_fixed(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    tp = fp = fn = 0
    for key, value in pred.items():
        if key not in gt:
            continue
        binary = (value.squeeze() > threshold).astype(np.uint8)
        pred_scenes = predictions_to_scenes(binary)
        _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt[key], pred_scenes)
        tp += tp_
        fp += fp_
        fn += fn_
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_temperature(logits: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> float:
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for key, arr in logits.items():
        if key not in gt:
            continue
        labels, _ = scenes2zero_one_representation(gt[key], len(arr))
        all_logits.append(torch.from_numpy(arr.astype(np.float32)))
        all_labels.append(torch.from_numpy(labels.astype(np.float32)).unsqueeze(1))

    if not all_logits:
        return 1.0

    logits_t = torch.cat(all_logits, 0)
    labels_t = torch.cat(all_labels, 0)

    def objective(temp: float) -> float:
        return float(F.binary_cross_entropy_with_logits(logits_t / temp, labels_t).item())

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def maybe_cap_keys(keys: list[str], limit: int) -> list[str]:
    if limit > 0:
        return keys[:limit]
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--base-ckpt", default=DEFAULT_BASE_CKPT)
    parser.add_argument("--out-ckpt", default=DEFAULT_OUT_CKPT)
    parser.add_argument("--sample-cache", default=DEFAULT_SAMPLE_CACHE)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--eval-cache-dir", default=DEFAULT_EVAL_CACHE_DIR)
    parser.add_argument("--resume-state", default=DEFAULT_RESUME_STATE)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--run-manifest", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--manyhot-weight", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--temperature-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--boundary-window", type=int, default=1)
    parser.add_argument("--max-samples-per-video", type=int, default=160)
    parser.add_argument("--max-total-samples", type=int, default=0)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--min-neg-per-video", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=200)
    parser.add_argument("--max-test-videos", type=int, default=0)
    parser.add_argument("--max-cache-video-frames", type=int, default=180000)
    parser.add_argument("--max-cache-video-seconds", type=float, default=7200.0)
    parser.add_argument("--save-every-videos", type=int, default=50)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--stop-after-minutes", type=float, default=0.0)
    parser.add_argument("--rebuild-sample-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--ignore-resume-config", action="store_true")
    parser.add_argument("--no-eval-cache", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    args = parser.parse_args()
    args._deadline = time.monotonic() + args.stop_after_minutes * 60.0 if args.stop_after_minutes > 0 else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    meta = load_metadata(args.meta)
    entries = meta["entries"]
    train_keys = list(meta["train_keys"])
    val_keys = maybe_cap_keys(list(meta["val_keys"]), args.max_val_videos)
    shot_test_entries = meta["shot_test_entries"]
    shot_test_keys = maybe_cap_keys(sorted(shot_test_entries.keys()), args.max_test_videos)
    print(f"Train keys: {len(train_keys)}  Val keys: {len(val_keys)}  Shot test keys: {len(shot_test_keys)}")

    base_hash = sha256_file(args.base_ckpt)
    backbone = load_supernet(args.base_ckpt)

    pretrained_head_cpu = {
        "fc1": {k: v.detach().cpu().clone() for k, v in backbone.fc1_0.state_dict().items()},
        "cls_layer1": {k: v.detach().cpu().clone() for k, v in backbone.cls_layer1.state_dict().items()},
        "cls_layer2": {k: v.detach().cpu().clone() for k, v in backbone.cls_layer2.state_dict().items()},
        "full_state": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
    }

    try:
        features, one_hot, boundary, sample_stats = build_or_load_sample_cache(
            args.sample_cache,
            args.meta,
            entries,
            train_keys,
            backbone,
            base_hash,
            args,
        )
    except TimeBudgetExpired as exc:
        print(f"{exc} Resume by rerunning the same command with previous outputs restored.")
        backbone.cpu()
        if device == "cuda":
            torch.cuda.empty_cache()
        return

    backbone.cpu()
    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Sample cache stats:")
    print(f"  samples: {sample_stats['samples']}")
    print(f"  one-hot positive rate: {sample_stats['one_hot_positive_rate']:.6f}")
    print(f"  boundary positive rate: {sample_stats['boundary_positive_rate']:.6f}")
    if sample_stats["skipped"]:
        print(f"  skipped videos: {len(sample_stats['skipped'])}")
    data_manifest = None
    if args.data_manifest:
        data_manifest = write_training_data_manifest(
            args.data_manifest,
            args.meta,
            args.base_ckpt,
            entries,
            sample_stats,
            args.data_seed,
        )
        print(f"Training data manifest -> {args.data_manifest}")

    dataset = SampleFeatureDataset(features, one_hot, boundary)
    in_features = int(features.shape[1])
    head = ClassificationHead(in_features=in_features).to(device)
    head.fc1.load_state_dict(pretrained_head_cpu["fc1"])
    head.cls_layer1.load_state_dict(pretrained_head_cpu["cls_layer1"])
    head.cls_layer2.load_state_dict(pretrained_head_cpu["cls_layer2"])

    training_started = time.perf_counter()
    try:
        head, train_losses = train_head(head, dataset, args)
    except TimeBudgetExpired as exc:
        print(f"{exc} Resume by rerunning the same command with previous outputs restored.")
        return
    training_elapsed_seconds = time.perf_counter() - training_started
    state_fingerprint = hash_head_state(head)
    full_state = merge_head_into_state(pretrained_head_cpu["full_state"], head)

    eval_model = load_model_from_state(full_state)
    os.makedirs(args.eval_cache_dir, exist_ok=True)

    val_cache_config = {
        "state_fingerprint": state_fingerprint,
        "keys_hash": hash_keys(val_keys),
        "split": "combined_val",
    }
    val_logits = load_or_run_logits(
        eval_model,
        entries,
        val_keys,
        os.path.join(args.eval_cache_dir, "combined_val_logits.pkl"),
        val_cache_config,
        args.no_eval_cache,
    )
    val_gt = gt_for_logits(entries, val_logits)

    pred_no_temp = logits_to_pred_dict(val_logits, temperature=1.0, sigma=args.sigma)
    val_no_temp = evaluate_best(pred_no_temp, val_gt)

    if args.temperature_mode == "auto":
        temperature = find_temperature(val_logits, val_gt)
        pred_temp = logits_to_pred_dict(val_logits, temperature=temperature, sigma=args.sigma)
        val_temp = evaluate_best(pred_temp, val_gt)
        use_temperature = val_temp["f1"] >= val_no_temp["f1"]
        deploy_temperature = temperature if use_temperature else 1.0
        deploy_threshold = val_temp["threshold"] if use_temperature else val_no_temp["threshold"]
        deploy_val = val_temp if use_temperature else val_no_temp
    else:
        temperature = 1.0
        val_temp = None
        use_temperature = False
        deploy_temperature = 1.0
        deploy_threshold = val_no_temp["threshold"]
        deploy_val = val_no_temp

    print("\nValidation:")
    print(f"  no temp: F1={val_no_temp['f1']:.6f} P={val_no_temp['precision']:.6f} R={val_no_temp['recall']:.6f} thr={val_no_temp['threshold']:.4f}")
    if val_temp is not None:
        print(f"  temp T={temperature:.4f}: F1={val_temp['f1']:.6f} P={val_temp['precision']:.6f} R={val_temp['recall']:.6f} thr={val_temp['threshold']:.4f}")
    else:
        print("  temp: disabled")
    print(f"  deploy: T={deploy_temperature:.4f} thr={deploy_threshold:.4f}")

    torch.save(
        {
            "net": full_state,
            "phase2_config": {
                "gamma": args.gamma,
                "alpha": args.alpha,
                "loss": args.loss,
                "temperature": deploy_temperature,
                "temperature_mode": args.temperature_mode,
                "use_temperature": use_temperature,
                "sigma": args.sigma,
                "threshold": deploy_threshold,
                "manyhot_weight": args.manyhot_weight,
                "boundary_window": args.boundary_window,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
                "val_f1": deploy_val["f1"],
                "training_data": "Shot train + ClipShots train",
                "metadata": os.path.abspath(args.meta),
                "sample_cache": os.path.abspath(args.sample_cache),
                "sample_cache_stats": sample_stats,
            },
        },
        args.out_ckpt,
    )
    print(f"\nCheckpoint saved -> {args.out_ckpt}")

    shot_test_best = None
    shot_test_deploy = None
    if not args.skip_test_eval:
        test_cache_config = {
            "state_fingerprint": state_fingerprint,
            "keys_hash": hash_keys(shot_test_keys),
            "split": "shot_test",
        }
        test_logits = load_or_run_logits(
            eval_model,
            shot_test_entries,
            shot_test_keys,
            os.path.join(args.eval_cache_dir, "shot_test_logits.pkl"),
            test_cache_config,
            args.no_eval_cache,
        )
        test_gt = gt_for_logits(shot_test_entries, test_logits)
        test_pred = logits_to_pred_dict(test_logits, temperature=deploy_temperature, sigma=args.sigma)
        shot_test_best = evaluate_best(test_pred, test_gt)
        shot_test_deploy = evaluate_fixed(test_pred, test_gt, deploy_threshold)
        print("\nShot test (200-video AutoShot test split):")
        print(f"  best sweep: F1={shot_test_best['f1']:.6f} P={shot_test_best['precision']:.6f} R={shot_test_best['recall']:.6f} thr={shot_test_best['threshold']:.4f}")
        print(f"  deploy thr={deploy_threshold:.4f}: F1={shot_test_deploy['f1']:.6f} P={shot_test_deploy['precision']:.6f} R={shot_test_deploy['recall']:.6f} TP={shot_test_deploy['tp']} FP={shot_test_deploy['fp']} FN={shot_test_deploy['fn']}")

    results = {
        "train_losses": train_losses,
        "training_elapsed_seconds_this_invocation": training_elapsed_seconds,
        "sample_stats": sample_stats,
        "val_no_temp": val_no_temp,
        "val_temp": val_temp,
        "deploy": {
            "temperature": deploy_temperature,
            "threshold": deploy_threshold,
            "use_temperature": use_temperature,
        },
        "shot_test_best": shot_test_best,
        "shot_test_deploy": shot_test_deploy,
        "args": vars(args),
    }
    with open(args.results, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved -> {args.results}")

    if args.run_manifest:
        run_manifest = {
            "schema_version": 1,
            "training_seed": args.seed,
            "data_seed": args.data_seed,
            "selected_keys_hash": sample_stats["selected_keys_hash"],
            "state_fingerprint": state_fingerprint,
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_")
            },
            "artifacts": {
                "checkpoint": {
                    "path": os.path.abspath(args.out_ckpt),
                    "sha256": sha256_file(args.out_ckpt),
                },
                "results": {
                    "path": os.path.abspath(args.results),
                    "sha256": sha256_file(args.results),
                },
                "data_manifest": (
                    {
                        "path": os.path.abspath(args.data_manifest),
                        "sha256": sha256_file(args.data_manifest),
                    }
                    if data_manifest is not None
                    else None
                ),
            },
            "validation": {
                "videos": len(val_logits),
                "logits_keys_hash": hash_keys(list(val_logits)),
                "no_temperature": val_no_temp,
                "temperature_candidate": val_temp,
            },
            "training_elapsed_seconds_this_invocation": training_elapsed_seconds,
            "test_evaluated": not args.skip_test_eval,
        }
        os.makedirs(os.path.dirname(args.run_manifest) or ".", exist_ok=True)
        with open(args.run_manifest, "w", encoding="utf-8") as handle:
            json.dump(run_manifest, handle, indent=2)
            handle.write("\n")
        print(f"Run manifest -> {args.run_manifest}")


if __name__ == "__main__":
    main()
