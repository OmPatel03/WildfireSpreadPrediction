"""Analyze fire pixel distribution across train/val/test sets and years."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import h5py
from dataloader.FireSpreadDataModule import FireSpreadDataModule


def analyze_fire_distribution(data_dir: str = "/u50/capstone/cs4zp6g17/data/hdf5", sample_rate: float = 1.0):
    """Analyze fire pixel distribution across all years and data splits.
    
    Args:
        data_dir: Path to HDF5 data directory
        sample_rate: Fraction of fires to sample (1.0 = all fires, 0.5 = 50% of fires for speed)
    """
    
    train_years, val_years, test_years = FireSpreadDataModule.split_fires(data_fold_id=0)
    
    print("=" * 80)
    print("FIRE PIXEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nData split (fold 0):")
    print(f"  Train years: {train_years}")
    print(f"  Val years:   {val_years}")
    print(f"  Test years:  {test_years}")
    print(f"\nSampling {sample_rate*100:.0f}% of fires for analysis...")
    print()
    
    # Analyze each year
    data_path = Path(data_dir)
    yearly_stats = {}
    
    for year in train_years + val_years + test_years:
        print(f"Processing {year}...", end=" ", flush=True)
        
        year_dir = data_path / str(year)
        if not year_dir.exists():
            print("(directory not found)")
            continue
        
        fire_files = sorted(year_dir.glob("fire_*.hdf5"))
        
        # Sample fires if needed
        if sample_rate < 1.0:
            n_sample = max(1, int(len(fire_files) * sample_rate))
            indices = np.random.choice(len(fire_files), size=n_sample, replace=False)
            fire_files = [fire_files[i] for i in sorted(indices)]
        
        fire_pixels = 0
        total_pixels = 0
        n_samples = 0
        
        for fire_file in fire_files:
            try:
                with h5py.File(fire_file, 'r') as f:
                    data = f["data"][:]  # Shape: (time, features, height, width)
                    
                    # Last feature of each time step is the fire mask
                    # Each sample predicts the next day's fire
                    # So we have len(data)-1 samples per fire
                    n_time_steps = data.shape[0]
                    
                    for t in range(n_time_steps - 1):
                        # Label is last feature of next time step
                        fire_mask = data[t + 1, -1, :, :]
                        fire_pixels += np.sum(fire_mask > 0.5)
                        total_pixels += fire_mask.size
                        n_samples += 1
            except Exception as e:
                print(f"\nError reading {fire_file}: {e}")
                continue
        
        if total_pixels > 0:
            fire_rate = fire_pixels / total_pixels
            yearly_stats[year] = {
                'fire_pixels': fire_pixels,
                'total_pixels': total_pixels,
                'fire_rate': fire_rate,
                'n_samples': n_samples,
                'n_fires_sampled': len(fire_files)
            }
            print(f"Fire rate: {fire_rate*100:.2f}% ({fire_pixels:,}/{total_pixels:,} pixels), {n_samples} samples")
        else:
            print("(no data)")
    
    print("\n" + "=" * 80)
    print("SUMMARY BY DATA SPLIT")
    print("=" * 80)
    
    splits = {
        'Train': train_years,
        'Val': val_years,
        'Test': test_years
    }
    
    for split_name, years in splits.items():
        total_fire_pixels = sum(yearly_stats[y]['fire_pixels'] for y in years if y in yearly_stats)
        total_all_pixels = sum(yearly_stats[y]['total_pixels'] for y in years if y in yearly_stats)
        total_samples = sum(yearly_stats[y]['n_samples'] for y in years if y in yearly_stats)
        
        fire_rate = total_fire_pixels / total_all_pixels if total_all_pixels > 0 else 0
        pos_class_weight = 1 / fire_rate if fire_rate > 0 else 1.0
        
        print(f"\n{split_name} Set (years {years}):")
        print(f"  Fire pixels:      {total_fire_pixels:>12,} ({fire_rate*100:>6.2f}%)")
        print(f"  Non-fire pixels:  {total_all_pixels - total_fire_pixels:>12,} ({100-fire_rate*100:>6.2f}%)")
        print(f"  Total pixels:     {total_all_pixels:>12,}")
        print(f"  Total samples:    {total_samples:>12,}")
        print(f"  pos_class_weight: {pos_class_weight:>12.1f}")
        
    print("\n" + "=" * 80)
    print("IMPLICATIONS FOR VQ-PRIORITY SAMPLING")
    print("=" * 80)
    print("""
Severe class imbalance (mostly non-fire pixels):
  ✗ Most clusters will be dominated by non-fire examples
  ✗ VQ-priority sampling may prioritize rare fire clusters
  ✗ But without proper balancing, model learns mostly non-fire features
  ✗ Validation metric (F1 on mixed pixels) becomes very low

Solutions already implemented:
  ✓ Monitor val_f1 instead of val_loss → optimizes for fire detection
  ✓ Use pos_class_weight in loss function → penalizes fire misclassification
  ✓ Increase min_epochs to 30 → allows model to adapt to data distribution
  ✓ Increase learning rate to 0.001 → faster convergence despite imbalance

Expected improvements:
  • F1 should improve as model learns to detect fires despite imbalance
  • VQ-priority sampling helps focus on hard-to-predict examples
  • Class-weighted loss compensates for pixel imbalance
""")


if __name__ == "__main__":
    analyze_fire_distribution()
