from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from autoshotv2.runtime import AutoShotRuntime


@lru_cache(maxsize=4)
def get_runtime(model_path: Path | None = None) -> AutoShotRuntime:
    settings = get_settings()
    path = model_path or settings.autoshot_model_path
    return AutoShotRuntime(path, settings.autoshot_device)


def clear_runtime_cache() -> None:
    get_runtime.cache_clear()
