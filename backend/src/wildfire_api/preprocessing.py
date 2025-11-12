from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as TF

from .config import Settings
from .wsts_bridge import ensure_wsts_on_path

ensure_wsts_on_path()
from dataloader.utils import (  # noqa: E402
    get_indices_of_degree_features,
    get_means_stds_missing_values,
)


@dataclass
class PreprocessedSample:
    tensor: torch.Tensor
    ground_truth: torch.Tensor
    sample_index: int
    total_samples: int
    spatial_shape: Tuple[int, int]


class SamplePreprocessor:
    def __init__(self, settings: Settings):
        self._settings = settings
        means, stds, _ = get_means_stds_missing_values(list(settings.stats_years))
        self._means = torch.tensor(means, dtype=torch.float32)[None, :, None, None]
        self._stds = torch.tensor(stds, dtype=torch.float32)[None, :, None, None]
        self._degree_indices = torch.tensor(
            get_indices_of_degree_features(), dtype=torch.long
        )
        self._one_hot = torch.eye(17, dtype=torch.float32)

    def prepare(self, cube: np.ndarray, sample_offset: int = -1) -> PreprocessedSample:
        time_steps = cube.shape[0]
        leads = self._settings.n_leading_observations
        total_samples = time_steps - leads
        if total_samples <= 0:
            raise ValueError(
                f"Fire contains {time_steps} frames, needs at least {leads + 1}."
            )
        index = self._resolve_sample_index(sample_offset, total_samples)
        start = index
        end = start + leads
        label_idx = end

        x_np = np.copy(cube[start:end])
        y_np = np.copy(cube[label_idx, -1, ...])

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        y = (y > 0).long()
        x, y = self._center_crop_x32(x, y)

        x = self._apply_trigonometric_encoding(x)
        binary_mask = (x[:, -1:, ...] > 0).float()
        x = self._standardize(x)
        x = torch.cat([x, binary_mask], dim=1)
        x = torch.nan_to_num(x, nan=0.0)
        x = self._expand_landcover(x)

        tensor = x.unsqueeze(0)

        _, _, height, width = x.shape
        return PreprocessedSample(
            tensor=tensor,
            ground_truth=y,
            sample_index=index,
            total_samples=total_samples,
            spatial_shape=(height, width),
        )

    def _resolve_sample_index(self, offset: int, total_samples: int) -> int:
        if offset < 0:
            offset = total_samples + offset
        if offset < 0 or offset >= total_samples:
            raise IndexError(
                f"Sample offset {offset} outside range [0, {total_samples - 1}]"
            )
        return offset

    def _center_crop_x32(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = x.shape
        new_h = (height // 32) * 32
        new_w = (width // 32) * 32
        if new_h == 0 or new_w == 0:
            raise ValueError(
                f"Cannot crop tensor of shape {(height, width)} down to multiples of 32."
            )
        x = TF.center_crop(x, (new_h, new_w))
        y = TF.center_crop(y, (new_h, new_w))
        return x, y

    def _apply_trigonometric_encoding(self, x: torch.Tensor) -> torch.Tensor:
        indices = self._degree_indices
        x[:, indices, ...] = torch.sin(torch.deg2rad(x[:, indices, ...]))
        return x

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._means) / self._stds

    def _expand_landcover(self, x: torch.Tensor) -> torch.Tensor:
        landcover = x[:, 16, ...].long()
        landcover = landcover.clamp(min=1)
        flattened = landcover.flatten() - 1
        flattened = flattened.clamp(min=0, max=self._one_hot.shape[0] - 1)
        encoding = self._one_hot[flattened].reshape(
            x.shape[0], x.shape[2], x.shape[3], -1
        )
        encoding = encoding.permute(0, 3, 1, 2)
        return torch.cat([x[:, :16, ...], encoding, x[:, 17:, ...]], dim=1)
