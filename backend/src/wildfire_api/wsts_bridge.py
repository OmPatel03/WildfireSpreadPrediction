from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_wsts_on_path() -> Path:
    """Expose the research code (wsts) to the runtime sys.path."""
    package_root = Path(__file__).resolve().parents[3]
    wsts_src = package_root / "src" / "wsts" / "src"
    if not wsts_src.exists():
        raise FileNotFoundError(
            f"Could not locate wsts sources at {wsts_src}. "
            "Check that the repository submodule is present."
        )
    path_str = str(wsts_src)
    if path_str not in sys.path:
        sys.path.append(path_str)
    return wsts_src
