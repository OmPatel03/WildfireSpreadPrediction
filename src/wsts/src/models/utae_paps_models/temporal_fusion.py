import torch
import torch.nn.functional as F


def relative_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(1, seq_len + 1, device=device, dtype=torch.long)
    return positions.unsqueeze(0).repeat(batch_size, 1)


def apply_attention_to_scale(
    feats: torch.Tensor,
    attn_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    _, channels, height, width = feats.shape

    feats = feats.reshape(batch_size, seq_len, channels, height, width)
    attn = attn_mask.mean(dim=0)

    if attn.shape[-2:] != (height, width):
        attn = F.interpolate(
            attn.reshape(batch_size * seq_len, 1, *attn.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        attn = attn.reshape(batch_size, seq_len, height, width)

    attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
    attn_expanded = attn.unsqueeze(2)
    fused = (feats * attn_expanded).sum(dim=1)
    return fused
