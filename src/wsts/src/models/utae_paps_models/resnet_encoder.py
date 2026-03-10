import torch.nn as nn
import segmentation_models_pytorch as smp


class SharedResNetEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        depth: int = 5,
        weights: str | None = "imagenet",
        ltae_channels: int = 128,
    ):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=depth,
            weights=weights,
        )

        self.out_channels = self.encoder.out_channels
        self.ltae_channels = ltae_channels
        self.ltae_feature_index = self.out_channels.index(self.ltae_channels)
        self.decoder_in_channels = self.out_channels[-1]

    def forward(self, x):
        return self.encoder(x)
