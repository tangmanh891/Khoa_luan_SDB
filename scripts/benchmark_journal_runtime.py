"""Benchmark decoding, inference, post-processing, and end-to-end runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from autoshotv2 import runtime


def measure(operation: Callable[[], object], rounds: int) -> list[float]:
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return values


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_seconds": float(statistics.median(values)),
        "mean_seconds": float(statistics.mean(values)),
        "min_seconds": float(min(values)),
        "max_seconds": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    device = runtime.resolve_device(args.device)
    model = runtime.load_model(args.checkpoint, device)
    frames = runtime.decode_video_frames(args.video)
    logits = runtime.predict_frame_logits(model, frames, device)

    for _ in range(args.warmup):
        runtime.predict_frame_logits(model, frames, device)
        runtime.logits_to_probabilities(logits, args.temperature, args.sigma)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    decode_times = measure(
        lambda: runtime.decode_video_frames(args.video),
        args.rounds,
    )
    inference_times = measure(
        lambda: runtime.predict_frame_logits(model, frames, device),
        args.rounds,
    )
    postprocess_times = measure(
        lambda: runtime.probabilities_to_scenes(
            runtime.logits_to_probabilities(
                logits,
                args.temperature,
                args.sigma,
            ),
            args.threshold,
        ),
        args.rounds,
    )

    def end_to_end() -> None:
        decoded = runtime.decode_video_frames(args.video)
        predicted = runtime.predict_frame_logits(model, decoded, device)
        probabilities = runtime.logits_to_probabilities(
            predicted,
            args.temperature,
            args.sigma,
        )
        runtime.probabilities_to_scenes(probabilities, args.threshold)

    end_to_end_times = measure(end_to_end, args.rounds)
    median_end_to_end = statistics.median(end_to_end_times)
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "video": str(args.video.resolve()),
        "device": device,
        "frames": int(len(frames)),
        "warmup_rounds": args.warmup,
        "measurement_rounds": args.rounds,
        "postprocess": {
            "temperature": args.temperature,
            "sigma": args.sigma,
            "threshold": args.threshold,
        },
        "decode": summary(decode_times),
        "inference": summary(inference_times),
        "postprocess_runtime": summary(postprocess_times),
        "end_to_end": {
            **summary(end_to_end_times),
            "frames_per_second": float(len(frames) / median_end_to_end),
        },
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Runtime benchmark -> {args.output}: "
        f"{payload['end_to_end']['frames_per_second']:.2f} FPS"
    )


if __name__ == "__main__":
    main()
