from fastapi import APIRouter

from app.core.config import get_settings
from app.db.mongo import get_database
from autoshotv2.runtime import resolve_device

router = APIRouter()

DEFAULT_PRESET = "autoshotv2"

PRESETS: dict[str, dict] = {
    "autoshotv2": {
        "display_name": "AutoShotV2",
        "filename": "autoshotv2.pth",
        "threshold": None,
    },
    "autoshotv2_heatmap": {
        "display_name": "AutoShot V2 HeatMap",
        "filename": "autoshotv2_heatmap.pth",
        "threshold": None,
    },
    "best_shot": {
        "display_name": "AutoShotV2-SHOT",
        "filename": "ckpt_phase2_shot_f1_sweep_best.pth",
        "threshold": 0.12,
    },
    "best_clipshot": {
        "display_name": "AutoShotV2-ClipShot",
        "filename": "ckpt_phase2_shot_f1_sweep_best.pth",
        "threshold": 0.19,
    },
    "best_bbc": {
        "display_name": "AutoShotV2-BBC",
        "filename": "autoshotv2.pth",
        "threshold": None,
    },
    "autoshot": {
        "display_name": "AutoShot",
        "filename": "autoshot.pth",
        "threshold": None,
    },
}


@router.get("/models")
async def list_models() -> dict:
    settings = get_settings()
    models_dir = settings.autoshot_models_dir
    available = [
        {
            "preset": key,
            "display_name": p["display_name"],
            "is_default": key == DEFAULT_PRESET,
            "available": (models_dir / p["filename"]).is_file(),
        }
        for key, p in PRESETS.items()
    ]
    return {"models": available, "default": DEFAULT_PRESET}


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    database_status = "ok"
    try:
        await get_database().command("ping")
    except Exception:
        database_status = "unavailable"

    checkpoint_exists = settings.autoshot_model_path.is_file()
    preferred_backend = (
        "baseline"
        if settings.autoshot_use_baseline or not checkpoint_exists
        else "phase2"
    )
    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "database": database_status,
        "model": {
            "checkpoint": str(settings.autoshot_model_path),
            "checkpoint_exists": checkpoint_exists,
            "requested_device": settings.autoshot_device,
            "effective_device": resolve_device(settings.autoshot_device),
            "preferred_backend": preferred_backend,
        },
    }
