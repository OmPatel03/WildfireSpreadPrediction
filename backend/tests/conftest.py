"""
Pytest configuration and shared fixtures for the wildfire API test suite.

IMPORTANT – sys.modules stubs must be registered BEFORE any wildfire_api
submodule is imported, because preprocessing.py and model_loader.py call
ensure_wsts_on_path() at *module* level and depend on wsts source packages
that are unavailable in a pure unit-test environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1.  Make backend/src importable
# ---------------------------------------------------------------------------
_BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

# ---------------------------------------------------------------------------
# 2.  Point env-vars at non-existent but path-safe locations so get_settings()
#     succeeds (it uses strict=False on Path.resolve).
# ---------------------------------------------------------------------------
os.environ.setdefault("MODEL_CHECKPOINT", "/tmp/fake_model.ckpt")
os.environ.setdefault("HDF5_ROOT", "/tmp/fake_hdf5")

# ---------------------------------------------------------------------------
# 3.  Stub out wsts / external ML packages before wildfire_api is imported
# ---------------------------------------------------------------------------

# wildfire_api.wsts_bridge  ──  ensure_wsts_on_path must be callable
_mock_bridge = MagicMock()
_mock_bridge.ensure_wsts_on_path = MagicMock(return_value=Path("/fake/wsts"))
sys.modules.setdefault("wildfire_api.wsts_bridge", _mock_bridge)

# dataloader package (used by preprocessing.py via wsts bridge)
_mock_dl_utils = MagicMock()
_mock_dl_utils.get_means_stds_missing_values.return_value = (
    [0.0] * 40,   # means
    [1.0] * 40,   # stds
    [0.0] * 40,   # fill / missing-value sentinels
)
_mock_dl_utils.get_indices_of_degree_features.return_value = []
sys.modules.setdefault("dataloader", MagicMock())
sys.modules.setdefault("dataloader.utils", _mock_dl_utils)

# models package (used by model_loader.py: `from models import SMPModel`)
sys.modules.setdefault("models", MagicMock())

# ---------------------------------------------------------------------------
# 4.  Now safe to import wildfire_api helpers (no wsts deps in these modules)
# ---------------------------------------------------------------------------
import numpy as np
import pytest
from fastapi.testclient import TestClient

from wildfire_api.domain import SpreadPrediction
from wildfire_api.geojson import build_geojson
from wildfire_api.repository import WildfireMetadata

# ---------------------------------------------------------------------------
# 5.  Shared fake data
# ---------------------------------------------------------------------------

FAKE_METADATA = WildfireMetadata(
    fire_id="fire_00001",
    year=2021,
    path=Path("/fake/2021/fire_00001.hdf5"),
    longitude=-120.457,
    latitude=38.892,
    time_steps=10,
    feature_count=40,
    height=64,
    width=64,
    samples=9,
)

# Probability array: background ≈ 0.3, hot patch in centre ≈ 0.8
_probs = np.full((64, 64), 0.3, dtype=np.float32)
_probs[20:40, 20:40] = 0.8
_mask = (_probs >= 0.5).astype(np.uint8)
_gt = np.zeros((64, 64), dtype=np.uint8)
_gt[25:35, 25:35] = 1  # 100 "burnt" pixels inside the hot patch

FAKE_PREDICTION = SpreadPrediction(
    metadata=FAKE_METADATA,
    sample_index=8,
    total_samples=9,
    threshold=0.5,
    probabilities=_probs,
    mask=_mask,
    ground_truth=_gt,
    observation_dates=("2021-10-01",),
    target_date="2021-10-02",
)

FAKE_GEOJSON = build_geojson(FAKE_PREDICTION)

HEALTH_DICT: dict = {
    "status": "ok",
    "model_path": "/app/resources/model.ckpt",
    "model_loaded": True,
    "device": "cpu",
    "hdf5_root": "/data/HDF5",
    "default_year": 2021,
}

# ---------------------------------------------------------------------------
# 6.  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_svc() -> MagicMock:
    """A session-scoped mock WildfireService with pre-wired return values."""
    svc = MagicMock()
    svc.health.return_value = HEALTH_DICT
    svc.catalog.return_value = [FAKE_METADATA]
    svc.find_spread.return_value = (FAKE_PREDICTION, FAKE_GEOJSON)
    return svc


@pytest.fixture(scope="session")
def test_app(mock_svc: MagicMock):
    """
    Build a FastAPI application whose service layer is fully mocked.

    Strategy
    --------
    main.py executes ``app = create_app()`` at module level.  We must ensure
    that WildfireService is patched *before* that import runs.  We:
      1. Remove any cached import of ``main`` so Python re-executes it.
      2. Clear the lru_cache on get_settings() so a fresh Settings object is built.
      3. Enter patch context managers that replace ``WildfireService`` in both
         the ``wildfire_api`` package namespace and the ``wildfire_api.service``
         module (both are consulted during ``from wildfire_api import WildfireService``).
      4. Import ``main`` – this triggers ``app = create_app()`` which calls
         ``WildfireService(settings)`` → receives ``mock_svc``.
      5. Return ``main.app`` (route closures all close over ``mock_svc``).
    """
    sys.modules.pop("main", None)

    from wildfire_api.config import get_settings
    get_settings.cache_clear()

    with (
        patch("wildfire_api.WildfireService", return_value=mock_svc),
        patch("wildfire_api.service.WildfireService", return_value=mock_svc),
    ):
        import main as _main  # triggers app = create_app() under the patch
        application = _main.app

    return application


@pytest.fixture(scope="session")
def client(test_app) -> TestClient:
    """Session-wide TestClient backed by the mocked application."""
    return TestClient(test_app, raise_server_exceptions=False)
