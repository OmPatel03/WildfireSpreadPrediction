import torch.nn as nn
from .utae import UpConvBlock



class UNetDecoder(nn.Module):
    def __init__(self, encoder_out_channels):
        super(UNetDecoder, self).__init__()
        # Fused feature scales expected as:
        # [fused_64_h2, fused_64_h4, fused_128_h8, fused_256_h16, fused_512_h32]
        # Use UTAE decoder blocks directly.
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=encoder_out_channels,
                    d_out=256,
                    d_skip=256,
                    k=4,
                    s=2,
                    p=1,
                    norm="batch",
                    padding_mode="reflect",
                ),
                UpConvBlock(
                    d_in=256,
                    d_out=128,
                    d_skip=128,
                    k=4,
                    s=2,
                    p=1,
                    norm="batch",
                    padding_mode="reflect",
                ),
                UpConvBlock(
                    d_in=128,
                    d_out=64,
                    d_skip=64,
                    k=4,
                    s=2,
                    p=1,
                    norm="batch",
                    padding_mode="reflect",
                ),
                UpConvBlock(
                    d_in=64,
                    d_out=64,
                    d_skip=64,
                    k=4,
                    s=2,
                    p=1,
                    norm="batch",
                    padding_mode="reflect",
                ),
            ]
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, fused_features):
        fused_64_h2, fused_64_h4, fused_128_h8, fused_256_h16, fused_512_h32 = fused_features

        decoded = self.up_blocks[0](fused_512_h32, fused_256_h16)
        decoded = self.up_blocks[1](decoded, fused_128_h8)
        decoded = self.up_blocks[2](decoded, fused_64_h4)
        decoded = self.up_blocks[3](decoded, fused_64_h2)
        decoded = self.final_upsample(decoded)

        return decoded

        