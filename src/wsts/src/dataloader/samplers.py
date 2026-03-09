"""Samplers for domain-adversarial training."""

from typing import Iterator, Optional, Sized
import torch
from torch.utils.data import Sampler, Dataset
import numpy as np


class YearBalancedSampler(Sampler):
    """
    Sampler that balances year distribution within each batch.
    
    Ensures that mini-batches contain roughly equal representation from each year,
    preventing the domain classifier from trivially predicting year based on 
    batch composition alone.
    
    Works by:
    1. Grouping samples by year
    2. Cycling through years to form balanced batches
    3. Shuffling within year groups for training
    
    Note: This sampler requires dataset to return year labels.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: Dataset that supports returning year labels (must have __len__)
            batch_size: Batch size used in DataLoader
            shuffle: Whether to shuffle samples within year groups
            drop_last: Whether to drop last partial batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group indices by year
        self.year_indices = self._group_by_year()
        self.n_years = len(self.year_indices)
        
        # Compute number of batches
        self.n_samples = len(dataset)
        if self.drop_last:
            self.n_batches = self.n_samples // self.batch_size
        else:
            self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def _group_by_year(self) -> dict:
        """
        Group dataset indices by year label.
        
        Returns:
            Dict mapping year -> list of indices for that year
        """
        year_indices = {}
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            # Handle different batch formats
            if isinstance(sample, tuple):
                if len(sample) == 3:
                    # (x, y, year)
                    year = sample[2].item() if hasattr(sample[2], 'item') else sample[2]
                elif len(sample) == 4:
                    # (x, y, doy, year)
                    year = sample[3].item() if hasattr(sample[3], 'item') else sample[3]
                else:
                    raise ValueError(f"Unexpected batch format with {len(sample)} elements")
            else:
                raise ValueError(f"Expected tuple batch, got {type(sample)}")
            
            if year not in year_indices:
                year_indices[year] = []
            year_indices[year].append(idx)
        
        # Shuffle within year groups if requested
        if self.shuffle:
            for year in year_indices:
                np.random.shuffle(year_indices[year])
        
        return year_indices
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for balanced batches."""
        # Create cycling iterators for each year
        year_iter = {
            year: iter(np.tile(indices, (self.n_batches, 1)).flatten())
            for year, indices in self.year_indices.items()
        }
        
        batch_indices = []
        samples_per_year = self.batch_size // self.n_years
        remainder = self.batch_size % self.n_years
        
        for batch_idx in range(self.n_batches):
            batch = []
            
            # Sample from each year
            for year_idx, year in enumerate(sorted(self.year_indices.keys())):
                # Slightly more samples for first 'remainder' years
                n_samples = samples_per_year + (1 if year_idx < remainder else 0)
                
                for _ in range(n_samples):
                    try:
                        batch.append(next(year_iter[year]))
                    except StopIteration():
                        # Reshufffffle and restart if we run out
                        year_samples = [i for i in self.year_indices[year]]
                        if self.shuffle:
                            np.random.shuffle(year_samples)
                        year_iter[year] = iter(np.tile(year_samples, (2, 1)).flatten())
                        batch.append(next(year_iter[year]))
            
            # Handle last batch if not dropping
            if len(batch) == self.batch_size:
                for idx in batch:
                    yield idx
            elif not self.drop_last:
                for idx in batch:
                    yield idx
    
    def __len__(self) -> int:
        """Return number of samples yielded by sampler."""
        if self.drop_last:
            return (self.n_samples // self.batch_size) * self.batch_size
        else:
            return self.n_samples
