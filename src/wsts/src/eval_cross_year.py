"""
Evaluation utilities for domain-invariant wildfire models.

Computes cross-year generalization metrics:
- Per-year AP, precision, recall
- Worst-year AP (minimum across years)
- Year-to-year standard deviation
- Year classifier accuracy (diagnostic)
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from torchmetrics import AveragePrecision, Accuracy
from pathlib import Path
import json


def compute_per_year_metrics(
    predictions: Dict[int, torch.Tensor],
    targets: Dict[int, torch.Tensor],
    year_labels: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, any]:
    """
    Compute per-year evaluation metrics.
    
    Args:
        predictions: Dict mapping year -> tensor of predictions (B, H, W) or (B,)
        targets: Dict mapping year -> tensor of targets (B, H, W) or (B,)
        year_labels: Dict mapping year -> predicted year labels for domain classifier eval
    
    Returns:
        Dict with per-year metrics and aggregated statistics
    """
    metrics = {}
    
    ap_metric = AveragePrecision(task='binary')
    
    per_year_ap = {}
    per_year_precision = {}
    per_year_recall = {}
    per_year_f1 = {}
    
    for year in sorted(predictions.keys()):
        preds = predictions[year].float()
        targs = targets[year].float()
        
        # Flatten if needed
        if preds.ndim > 1:
            preds = preds.flatten()
            targs = targs.flatten()
        
        # Compute AP
        ap = ap_metric(preds, targs)
        per_year_ap[year] = ap.item()
        
        # Compute precision, recall, F1
        preds_binary = (preds > 0.5).long()
        tp = ((preds_binary == 1) & (targs == 1)).sum()
        fp = ((preds_binary == 1) & (targs == 0)).sum()
        fn = ((preds_binary == 0) & (targs == 1)).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        per_year_precision[year] = precision.item()
        per_year_recall[year] = recall.item()
        per_year_f1[year] = f1.item()
    
    # Aggregate statistics
    ap_values = list(per_year_ap.values())
    
    metrics['per_year'] = {
        'ap': per_year_ap,
        'precision': per_year_precision,
        'recall': per_year_recall,
        'f1': per_year_f1,
    }
    
    metrics['aggregate'] = {
        'mean_ap': float(np.mean(ap_values)),
        'std_ap': float(np.std(ap_values)),
        'worst_year_ap': float(np.min(ap_values)),
        'best_year_ap': float(np.max(ap_values)),
        'worst_year': int(sorted(per_year_ap.keys())[np.argmin(ap_values)]),
    }
    
    # Domain classifier accuracy (if provided)
    if year_labels:
        domain_accuracies = {}
        accuracy_metric = Accuracy(task='multiclass', num_classes=8)  # 2016-2023
        
        for year in year_labels:
            pred_years = year_labels[year]
            # Convert to 0-7 range
            true_years = torch.full_like(pred_years, year - 2016)
            acc = accuracy_metric(pred_years, true_years)
            domain_accuracies[year] = acc.item()
        
        metrics['domain_classifier'] = {
            'per_year_accuracy': domain_accuracies,
            'mean_accuracy': float(np.mean(list(domain_accuracies.values()))),
        }
    
    return metrics


def compute_transfer_matrix(
    results: List[Dict],
    metric_name: str = 'ap',
) -> np.ndarray:
    """
    Compute transfer learning matrix: train_year x test_year.
    
    Args:
        results: List of dicts with keys 'train_year', 'test_year', 'metrics'
        metric_name: Metric to extract (e.g., 'ap', 'f1')
    
    Returns:
        Matrix where [i,j] = performance training on year i, testing on year j
    """
    years = sorted(set([r['train_year'] for r in results] + [r['test_year'] for r in results]))
    n_years = len(years)
    year_to_idx = {y: i for i, y in enumerate(years)}
    
    matrix = np.zeros((n_years, n_years))
    
    for result in results:
        train_idx = year_to_idx[result['train_year']]
        test_idx = year_to_idx[result['test_year']]
        matrix[train_idx, test_idx] = result['metrics'].get(metric_name, 0.0)
    
    return matrix, years


def print_evaluation_report(metrics: Dict, name: str = "Evaluation"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{name} Results")
    print(f"{'='*60}\n")
    
    # Per-year metrics
    if 'per_year' in metrics:
        print("Per-Year Metrics:")
        print(f"{'Year':<10} {'AP':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
        print("-" * 52)
        
        for year in sorted(metrics['per_year']['ap'].keys()):
            ap = metrics['per_year']['ap'][year]
            prec = metrics['per_year']['precision'][year]
            rec = metrics['per_year']['recall'][year]
            f1 = metrics['per_year']['f1'][year]
            print(f"{year:<10} {ap:<10.4f} {prec:<12.4f} {rec:<10.4f} {f1:<10.4f}")
    
    # Aggregate metrics
    if 'aggregate' in metrics:
        print("\nAggregate Metrics:")
        agg = metrics['aggregate']
        print(f"Mean AP: {agg['mean_ap']:.4f}")
        print(f"Std AP:  {agg['std_ap']:.4f}")
        print(f"Worst-Year AP: {agg['worst_year_ap']:.4f} (year {agg['worst_year']})")
        print(f"Best-Year AP:  {agg['best_year_ap']:.4f}")
    
    # Domain classifier accuracy
    if 'domain_classifier' in metrics:
        print("\nDomain Classifier Accuracy (lower is better for domain invariance):")
        dc = metrics['domain_classifier']
        print(f"Mean Accuracy: {dc['mean_accuracy']:.4f}")
        print("Per-Year Accuracy:")
        for year in sorted(dc['per_year_accuracy'].keys()):
            print(f"  {year}: {dc['per_year_accuracy'][year]:.4f}")
    
    print(f"\n{'='*60}\n")


def save_metrics(metrics: Dict, output_path: Path):
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy types to native Python for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_types(metrics), f, indent=2)
    
    print(f"Metrics saved to {output_path}")
