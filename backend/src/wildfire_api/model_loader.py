from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch

from .config import Settings
from .wsts_bridge import ensure_wsts_on_path

ensure_wsts_on_path()
from models import SMPModel  # noqa: E402


MODEL_HPARAM_KEYS = {
    "encoder_name",
    "n_channels",
    "flatten_temporal_dimension",
    "pos_class_weight",
    "loss_function",
    "use_doy",
    "required_img_size",
}


class WildfireModel:
    def __init__(self, settings: Settings):
        self._settings = settings
        requested_device = os.getenv("WILDFIRE_DEVICE")
        if requested_device:
            self._device = torch.device(requested_device)
        else:
            self._device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
        checkpoint_path = str(settings.model_path)
        self._model = self._load_model_from_checkpoint(checkpoint_path)
        self._model.eval()
        self._model.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            batch = inputs.to(self._device)
            logits = self._model(batch)
            return torch.sigmoid(logits).cpu()

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> SMPModel:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hyper_params = checkpoint.get("hyper_parameters", {})
        init_args: Dict[str, Any] = {}

        for key in MODEL_HPARAM_KEYS:
            if key in hyper_params:
                init_args[key] = hyper_params[key]

        # allow overriding flattening behavior from settings if provided
        init_args.setdefault(
            "flatten_temporal_dimension", self._settings.flatten_temporal_dimension
        )

        model = SMPModel(**init_args)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        return model


_MODEL_CACHE: Dict[Tuple[str, str], WildfireModel] = {}


def get_model(settings: Settings) -> WildfireModel:
    """
    Returns a cached WildfireModel based on the checkpoint path and device selection
    derived from environment settings. Settings instances are not hashable, so we build
    an explicit cache keyed by (model_path, device).
    """
    checkpoint = str(settings.model_path.resolve())
    requested_device = os.getenv("WILDFIRE_DEVICE")
    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (checkpoint, device)

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WildfireModel(settings)
    return _MODEL_CACHE[cache_key]
