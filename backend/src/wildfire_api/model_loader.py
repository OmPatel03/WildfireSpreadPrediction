from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Tuple

import torch

from .config import Settings
from .wsts_bridge import ensure_wsts_on_path

ensure_wsts_on_path()
import models  # noqa: E402
from models import ResNet18UTAELightning, DomainAdversarialUTAELightning  # noqa: E402


MODEL_HPARAM_KEYS = {
    "encoder_name",
    "n_channels",
    "flatten_temporal_dimension",
    "pos_class_weight",
    "loss_function",
    "use_doy",
    "required_img_size",
}

LEGACY_OPTIONAL_STATE_PREFIXES = (
    "fire_reconstruction_head.",
)


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

    def _resolve_model_class(self, hyper_params: Dict[str, Any]) -> type:
        class_path = hyper_params.get("_class_path")
        if not class_path:
            return DomainAdversarialUTAELightning

        class_name = str(class_path).split(".")[-1]
        model_class = getattr(models, class_name, None)
        if model_class is None:
            raise ValueError(
                f"Unsupported checkpoint model class '{class_path}'. "
                f"Available classes are defined in the wsts models package."
            )
        return model_class

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> DomainAdversarialUTAELightning:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loading model from checkpoint: {checkpoint_path}")
        hyper_params = checkpoint.get("hyper_parameters", {})
        model_class = self._resolve_model_class(hyper_params)
        init_args: Dict[str, Any] = {}

        valid_keys = set(inspect.signature(model_class.__init__).parameters)
        valid_keys.discard("self")

        for key, value in hyper_params.items():
            if key in MODEL_HPARAM_KEYS or key in valid_keys:
                init_args[key] = value

        # allow overriding flattening behavior from settings if provided
        init_args.setdefault(
            "flatten_temporal_dimension", self._settings.flatten_temporal_dimension
        )

        model = model_class(**init_args)
        state_dict = checkpoint["state_dict"]
        filtered_state_dict = dict(state_dict)
        stripped_keys = [
            key
            for key in state_dict
            if key.startswith(LEGACY_OPTIONAL_STATE_PREFIXES)
        ]
        for key in stripped_keys:
            filtered_state_dict.pop(key, None)
        if stripped_keys:
            print(
                "Ignoring legacy checkpoint keys not used by the inference model: "
                + ", ".join(sorted(stripped_keys))
            )

        missing_keys, unexpected_keys = model.load_state_dict(
            filtered_state_dict, strict=False
        )
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch after compatibility filtering. "
                f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
            )
        print("Model loaded successfully.")
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
