"""Lightweight transform-domain blocks inspired by TD-FusionUNet."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class TransformDomainFusionBlock(nn.Module):
    """
    Residual bottleneck block that mixes spatial, DCT, and Hadamard views.

    This is a lightweight prototype rather than a full TD-FusionUNet rewrite.
    It lets us test whether simple transform-domain mixing on the fused temporal
    feature map helps AP in the existing domain-adversarial UTAE pipeline.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        max_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.max_residual_scale = float(max_residual_scale)

        self.input_norm = nn.BatchNorm2d(channels)
        self.frequency_mixer = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_project = nn.Sequential(
            nn.Conv2d(3 * channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self._dct_cache: Dict[Tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}
        self._hadamard_cache: Dict[
            Tuple[int, str, Optional[int], torch.dtype], Optional[torch.Tensor]
        ] = {}

    def _cache_key(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[int, str, Optional[int], torch.dtype]:
        return (n, device.type, device.index, dtype)

    def _get_dct_matrix(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = self._cache_key(n, device, dtype)
        cached = self._dct_cache.get(key)
        if cached is not None:
            return cached

        indices = torch.arange(n, device=device, dtype=dtype)
        mat = torch.cos(
            math.pi / n * (indices.unsqueeze(1) + 0.5) * indices.unsqueeze(0)
        )
        mat[0] *= math.sqrt(1.0 / n)
        if n > 1:
            mat[1:] *= math.sqrt(2.0 / n)
        self._dct_cache[key] = mat
        return mat

    def _get_hadamard_matrix(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        key = self._cache_key(n, device, dtype)
        cached = self._hadamard_cache.get(key)
        if key in self._hadamard_cache:
            return cached

        if n <= 0 or (n & (n - 1)) != 0:
            self._hadamard_cache[key] = None
            return None

        mat = torch.ones((1, 1), device=device, dtype=dtype)
        while mat.shape[0] < n:
            top = torch.cat([mat, mat], dim=1)
            bottom = torch.cat([mat, -mat], dim=1)
            mat = torch.cat([top, bottom], dim=0)
        mat = mat / math.sqrt(n)
        self._hadamard_cache[key] = mat
        return mat

    def _separable_transform(
        self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ij,bcjk,lk->bcil", left, x, right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x_norm = self.input_norm(x)

        dct_h = self._get_dct_matrix(height, x.device, x.dtype)
        dct_w = self._get_dct_matrix(width, x.device, x.dtype)
        dct_coeff = self._separable_transform(x_norm, dct_h, dct_w)

        had_h = self._get_hadamard_matrix(height, x.device, x.dtype)
        had_w = self._get_hadamard_matrix(width, x.device, x.dtype)
        if had_h is None or had_w is None:
            had_coeff = dct_coeff
        else:
            had_coeff = self._separable_transform(x_norm, had_h, had_w)

        mixed_coeffs = self.frequency_mixer(torch.cat([dct_coeff, had_coeff], dim=1))
        dct_coeff, had_coeff = torch.chunk(mixed_coeffs, 2, dim=1)

        dct_spatial = self._separable_transform(dct_coeff, dct_h.transpose(0, 1), dct_w.transpose(0, 1))
        if had_h is None or had_w is None:
            had_spatial = dct_spatial
        else:
            had_spatial = self._separable_transform(
                had_coeff, had_h.transpose(0, 1), had_w.transpose(0, 1)
            )

        fused = self.spatial_project(torch.cat([x_norm, dct_spatial, had_spatial], dim=1))
        residual_scale = self.max_residual_scale * torch.tanh(self.residual_scale)
        return x + residual_scale * fused
