from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    recon: torch.Tensor
    z_e: torch.Tensor
    z_q: torch.Tensor
    code_indices: torch.Tensor
    vq_loss: torch.Tensor
    recon_loss: torch.Tensor


class VectorQuantizerEMA(nn.Module):
    """Vector quantizer with EMA codebook updates."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.embedding = nn.Parameter(embed)
        self.embedding.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize z_e to nearest codebook vector.

        Returns:
            z_q: quantized tensor with straight-through gradients
            vq_loss: codebook + commitment loss
            encoding_indices: indices of selected codes
        """
        flat_inputs = z_e.reshape(-1, self.embedding_dim)

        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)

        z_q = torch.matmul(encodings, self.embedding).view_as(z_e)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay
                )

                dw = encodings.t() @ flat_inputs
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )

                self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + commitment_loss

        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, encoding_indices


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.mean(dim=(2, 3))
        return x


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        out_channels: int,
        hidden_channels: int = 256,
        init_resolution: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.init_resolution = init_resolution
        self.fc = nn.Linear(embedding_dim, hidden_channels * init_resolution * init_resolution)

        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, output_hw: Tuple[int, int]) -> torch.Tensor:
        b = z.shape[0]
        x = self.fc(z)
        x = x.view(b, self.hidden_channels, self.init_resolution, self.init_resolution)

        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.block2(x)
        x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x


class VQVAE(nn.Module):
    """VQ-VAE module for context clustering on WSTS features."""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 256,
        num_embeddings: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        flatten_temporal: bool = True,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.flatten_temporal = flatten_temporal

        self.encoder = encoder if encoder is not None else SimpleEncoder(in_channels, embedding_dim)
        self.codebook = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
        )
        self.decoder = decoder if decoder is not None else SimpleDecoder(embedding_dim, in_channels)

    def _maybe_flatten_temporal(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and self.flatten_temporal:
            b, t, c, h, w = x.shape
            return x.reshape(b, t * c, h, w)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_flatten_temporal(x)
        z_e = self.encoder(x)
        if z_e.dim() == 4:
            z_e = z_e.mean(dim=(2, 3))
        elif z_e.dim() == 3:
            z_e = z_e.mean(dim=2)
        return z_e

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.codebook(z_e)

    def decode(self, z_q: torch.Tensor, output_hw: Tuple[int, int]) -> torch.Tensor:
        return self.decoder(z_q, output_hw)

    def forward(self, x: torch.Tensor) -> VQOutput:
        x_flat = self._maybe_flatten_temporal(x)
        b, c, h, w = x_flat.shape

        z_e = self.encode(x_flat)
        z_q, vq_loss, code_indices = self.codebook(z_e)
        recon = self.decoder(z_q, (h, w))

        recon_loss = F.mse_loss(recon, x_flat)

        return VQOutput(
            recon=recon,
            z_e=z_e,
            z_q=z_q,
            code_indices=code_indices.view(b, -1).squeeze(1),
            vq_loss=vq_loss,
            recon_loss=recon_loss,
        )

    @torch.no_grad()
    def get_code_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Return codebook indices for each sample in the batch."""
        z_e = self.encode(x)
        _, _, indices = self.codebook(z_e)
        b = x.shape[0]
        return indices.view(b, -1).squeeze(1)