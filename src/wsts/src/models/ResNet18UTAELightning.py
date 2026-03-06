from typing import Any

import torch
import torch.nn as nn

from .BaseModel import BaseModel
from .utae_paps_models.ltae import LTAE2d
from .utae_paps_models.resnet_encoder import SharedResNetEncoder
from .utae_paps_models.temporal_fusion import apply_attention_to_scale, relative_positions
from .utae_paps_models.unet_decoder import UNetDecoder


class ResNet18UTAELightning(BaseModel):
    """_summary_ U-Net architecture with temporal attention and segmentation head
    """
    def __init__(
        self, 
        n_channels: int, 
        flatten_temporal_dimension: bool, 
        pos_class_weight: float,
        loss_function: str,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
        ltae_channels: int = 128,
        d_model: int = 256,
        n_head: int = 16,
        *args: Any, 
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=False,
            *args,
            **kwargs
        )
        self.model = self
        self.save_hyperparameters()

        # Shared pretrained ResNet encoder (ImageNet by default)
        self.shared_encoder = SharedResNetEncoder(
            encoder_name,
            in_channels=n_channels,
            depth=5,
            weights=encoder_weights,
            ltae_channels=ltae_channels,
        )
        # Backward-compatible alias.
        self.encoder = self.shared_encoder
        self.encoder_out_channels_for_ltae = self.shared_encoder.ltae_channels
        self.ltae_feature_index = self.shared_encoder.ltae_feature_index
        decoder_in_channels = self.shared_encoder.decoder_in_channels

        # Temporal Attention Encoder (LTAE2d)
        self.temporal_encoder = LTAE2d(
            in_channels=self.encoder_out_channels_for_ltae,
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, self.encoder_out_channels_for_ltae],
            return_att=True,
            d_k=4,
        )

        # U-Net Decoder (5 upsamples)
        self.decoder = UNetDecoder(decoder_in_channels)

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) - temporal sequence
            doys: ignored for this model (kept for interface compatibility)
        Returns:
            (B, 1, H, W) - fire prediction
        """
        batch_size, seq_len, channels, height, width = x.shape

        # Encoder returns list of feature maps at different resolutions
        x_flat = x.reshape(batch_size * seq_len, channels, height, width)
        feats_list = self.encoder(x_flat)  # List: [feat0, feat1, ..., feat5]
        
        # Compute LTAE attention from H/8 feature map (128 channels)
        ltae_source = feats_list[self.ltae_feature_index]
        _, ltae_channels, ltae_h, ltae_w = ltae_source.shape
        ltae_temporal = ltae_source.reshape(
            batch_size, seq_len, ltae_channels, ltae_h, ltae_w
        )

        batch_positions = relative_positions(batch_size, seq_len, x.device)

        ltae_fused, attn_mask = self.temporal_encoder(
            ltae_temporal,
            batch_positions=batch_positions,
            pad_mask=None,
        )
        # attn_mask shape: (n_head, B, T, h, w)
        fused_feats = []
        for feat_index, feat in enumerate(feats_list[1:], start=1):  # Skip feat0 (input layer)
            if feat_index == self.ltae_feature_index:
                # Use LTAE's own fused output at the attention-computation scale.
                fused_feat = ltae_fused
            else:
                fused_feat = apply_attention_to_scale(
                    feat,
                    attn_mask,
                    batch_size,
                    seq_len,
                )
            fused_feats.append(fused_feat)
        
        decoded = self.decoder(fused_feats)
        output = self.segmentation_head(decoded)

        return output