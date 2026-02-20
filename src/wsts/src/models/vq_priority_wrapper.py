from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .vq_vae import VQVAE, VQOutput


class VQPriorityWrapper(nn.Module):
    """Wrap a segmentation model with a VQ-VAE context encoder."""

    def __init__(
        self,
        segmentation_model: nn.Module,
        vqvae: VQVAE,
        feature_extractor: Optional[
            Callable[[nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
        ] = None,
        encoder_feature_index: int = -1,
    ) -> None:
        super().__init__()
        self.segmentation_model = segmentation_model
        self.vqvae = vqvae
        self.feature_extractor = feature_extractor
        self.encoder_feature_index = encoder_feature_index

    def _extract_bottleneck(
        self, x: torch.Tensor, doys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.feature_extractor is not None:
            return self.feature_extractor(self.segmentation_model, x, doys)

        if all(
            hasattr(self.segmentation_model, attr)
            for attr in ("encoder", "decoder", "segmentation_head")
        ):
            features = self.segmentation_model.encoder(x)
            if isinstance(features, (list, tuple)):
                return features[self.encoder_feature_index]
            return features

        raise ValueError(
            "No feature extractor provided and segmentation model does not expose an encoder."
        )

    def _forward_segmentation(
        self, x: torch.Tensor, doys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if all(
            hasattr(self.segmentation_model, attr)
            for attr in ("encoder", "decoder", "segmentation_head")
        ):
            features = self.segmentation_model.encoder(x)
            if isinstance(features, (list, tuple)):
                decoder_out = self.segmentation_model.decoder(*features)
            else:
                decoder_out = self.segmentation_model.decoder(features)
            return self.segmentation_model.segmentation_head(decoder_out)

        return self.segmentation_model(x, doys) if doys is not None else self.segmentation_model(x)

    def forward(
        self, x: torch.Tensor, doys: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, VQOutput]:
        seg_logits = self._forward_segmentation(x, doys)
        bottleneck = self._extract_bottleneck(x, doys)
        vq_out = self.vqvae(bottleneck)
        return seg_logits, vq_out

    @torch.no_grad()
    def get_code_indices(
        self, x: torch.Tensor, doys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bottleneck = self._extract_bottleneck(x, doys)
        return self.vqvae.get_code_indices(bottleneck)