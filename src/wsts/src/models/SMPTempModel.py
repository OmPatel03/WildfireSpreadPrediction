from typing import Any

import segmentation_models_pytorch as smp
import torch

from .BaseModel import BaseModel
from .utae_paps_models.ltae import LTAE2d
from .utae_paps_models.utae import Temporal_Aggregator


class SMPTempModel(BaseModel):
    """Paper-faithful Res18-UTAE baseline from the authors' public repo.

    This model keeps the SMP U-Net encoder/decoder and adds LTAE-based temporal
    fusion over encoder features, matching the released "Res18-UTAE" weights.
    """

    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str,
        encoder_weights: str | None = "imagenet",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=False,
            *args,
            **kwargs,
        )
        # Only record subclass-specific knobs here. Calling save_hyperparameters()
        # with no args would overwrite the normalized focal alpha stored by
        # BaseModel back to the raw class weight (e.g. 236), which breaks
        # torchvision's sigmoid_focal_loss.
        self.save_hyperparameters("encoder_name", "encoder_weights")

        encoder_weights = None if encoder_weights == "none" else encoder_weights
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=1,
        )
        self.last_stage_channels = self.model.encoder.out_channels[-1]
        self.ltae = LTAE2d(
            in_channels=self.last_stage_channels,
            n_head=16,
            d_k=4,
            mlp=[256, self.last_stage_channels],
            dropout=0.2,
            d_model=256,
            T=1000,
            return_att=True,
            positional_encoding=True,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="att_group")

    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _, _, _ = x.shape
        # The paper replaces day-of-year encoding with absolute sequence
        # positions 1..T for time-series input.
        positions = torch.arange(1, seq_len + 1, device=x.device).unsqueeze(0)
        positions = positions.repeat(batch_size, 1)

        num_stages = len(self.model.encoder.out_channels)
        encoder_features: list[list[torch.Tensor]] = [[] for _ in range(num_stages)]

        for t in range(seq_len):
            features = self.model.encoder(x[:, t, :, :, :])
            for i in range(num_stages):
                encoder_features[i].append(features[i])

        last_stage = torch.stack(encoder_features[-1], dim=1)
        aggregated_last, attn = self.ltae(last_stage, batch_positions=positions)

        aggregated_skips = []
        for i in range(1, num_stages - 1):
            stage = torch.stack(encoder_features[i], dim=1)
            aggregated = self.temporal_aggregator(stage, attn_mask=attn)
            aggregated_skips.append(aggregated)

        dummy = encoder_features[0][0]
        decoder_features = [dummy] + aggregated_skips + [aggregated_last]
        decoder_output = self.model.decoder(*decoder_features)
        masks = self.model.segmentation_head(decoder_output)
        return masks

    def load_state_dict(self, state_dict, strict: bool = True):
        conv1_key = "model.encoder.conv1.weight"
        if conv1_key in state_dict:
            pretrained_weight = state_dict[conv1_key]
            current_weight = self.state_dict()[conv1_key]
            if pretrained_weight.shape[1] != current_weight.shape[1]:
                factor = current_weight.shape[1] // pretrained_weight.shape[1]
                adapted_weight = pretrained_weight.repeat(1, factor, 1, 1) / factor
                state_dict[conv1_key] = adapted_weight
        return super().load_state_dict(state_dict, strict=strict)
