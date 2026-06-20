from argparse import Namespace
from pathlib import Path

from autoshotv2 import ablation


def runner_args(**overrides):
    values = {
        "force": False,
        "epochs": 3,
        "batch_size": 256,
        "max_samples_per_video": 160,
        "max_total_samples": 0,
        "neg_per_pos": 3,
        "min_neg_per_video": 32,
        "seed": 42,
        "data_seed": 42,
        "max_train_videos": 400,
        "max_val_videos": 50,
        "max_test_videos": 0,
        "save_every_videos": 25,
        "save_every_epochs": 1,
        "log_every_batches": 50,
        "stop_after_minutes": 0.0,
        "max_cache_video_frames": 180000,
        "max_cache_video_seconds": 7200.0,
        "no_eval_cache": False,
        "skip_test_eval": False,
        "rebuild_sample_cache": False,
        "resume_training": True,
        "continue_on_error": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_train_run_forwards_cache_budget_flags(tmp_path, monkeypatch):
    """The subprocess must see the same cache budgets that sample_cache_matches
    hashes into the expected cache identity, or the two silently diverge."""
    captured = {}

    def fake_run_command(cmd, cwd, continue_on_error):
        captured["cmd"] = cmd
        return True, ""

    monkeypatch.setattr(ablation, "run_command", fake_run_command)

    exp = ablation.Experiment(
        experiment_id="A1_phase2_bce_onehot",
        description="test",
        kind="phase2",
    )
    args = runner_args(max_cache_video_frames=90000, max_cache_video_seconds=1800.0)
    ok, _ = ablation.train_run(
        exp,
        tmp_path / "run",
        Path("."),
        tmp_path / "meta.pickle",
        tmp_path / "base.pth",
        tmp_path / "samples.pkl",
        args,
    )

    assert ok
    cmd = captured["cmd"]
    assert cmd[cmd.index("--max-cache-video-frames") + 1] == "90000"
    assert cmd[cmd.index("--max-cache-video-seconds") + 1] == "1800.0"
    assert cmd[cmd.index("--data-seed") + 1] == "42"
