from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.vq_priority_wrapper import VQPriorityWrapper


@torch.no_grad()
def compute_cluster_ids(
    vq_wrapper: VQPriorityWrapper,
    dataloader: DataLoader,
    device: torch.device,
    flatten_temporal: bool = False,
) -> torch.Tensor:
    """Compute cluster IDs for all samples in a dataloader."""
    vq_wrapper.eval()
    vq_wrapper.to(device)

    all_ids = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, _, doys = batch
        else:
            x, _ = batch
            doys = None

        if flatten_temporal and x.dim() == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        x = x.to(device)
        if doys is not None:
            doys = doys.to(device)

        batch_ids = vq_wrapper.get_code_indices(x, doys)
        all_ids.append(batch_ids.cpu())

    return torch.cat(all_ids, dim=0)


def compute_sample_weights(
    cluster_ids: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    """Inverse-frequency weights for each sample."""
    counts = torch.bincount(cluster_ids, minlength=num_embeddings).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts[cluster_ids]
    return weights


def compute_fire_density_weights(
    dataset,
    fire_threshold: float = 0.01,
) -> torch.Tensor:
    """Compute sample weights based on fire pixel density.
    
    Samples with fire pixels above the threshold get higher weight.
    This encourages the model to see more fire examples during training.
    
    Args:
        dataset: PyTorch dataset with (x, y) samples where y is fire ground truth
        fire_threshold: Fire pixel percentage threshold for upweighting (default 1%)
    
    Returns:
        Weight tensor of shape (num_samples,)
    """
    weights = []
    for i in range(len(dataset)):
        try:
            _, y = dataset[i]
            # y is fire ground truth, compute fire percentage
            fire_rate = y.float().mean().item()
            
            # Upweight samples with fire, downweight those without
            if fire_rate > fire_threshold:
                weight = fire_rate / fire_threshold  # Scale by how much above threshold
            else:
                weight = fire_threshold / (fire_threshold + 1)  # Downweight but don't eliminate
            
            weights.append(weight)
        except Exception as e:
            print(f"Warning: Could not compute weight for sample {i}: {e}")
            weights.append(1.0)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights to sum to number of samples (no change in effective batch size)
    weights = weights / weights.mean()
    
    return weights


def build_weighted_sampler(weights: torch.Tensor) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for oversampling rare clusters."""
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )