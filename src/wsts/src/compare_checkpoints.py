import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import ticker
import shutil
from tqdm import tqdm
import re
from pathlib import Path

from models import SMPModel
from dataloader.FireSpreadDataModule import FireSpreadDataModule


def get_checkpoint_label_from_path(ckpt_path):
    """Extract checkpoint label from path.
    
    Extracts the epoch number and run ID from checkpoint path.
    Example: './lightning_logs/wildfire_progression/mt00kgq8/checkpoints/epoch=58-step=4602.ckpt'
    Returns: 'Run mt00kgq8 (epoch=58)'
    """
    # Get the filename
    filename = Path(ckpt_path).name
    
    # Extract epoch from filename (epoch=58-step=4602.ckpt)
    epoch_match = re.search(r'epoch=(\d+)', filename)
    epoch = epoch_match.group(1) if epoch_match else "unknown"
    
    # Extract run ID from path (look for the directory between 'wildfire_progression' and 'checkpoints')
    path_parts = ckpt_path.replace('\\', '/').split('/')
    run_id = None
    for i, part in enumerate(path_parts):
        if part == 'wildfire_progression' and i + 1 < len(path_parts):
            run_id = path_parts[i + 1]
            break
    
    if run_id:
        return f"Run {run_id} (epoch={epoch})"
    else:
        return f"epoch={epoch}"


# Checkpoint paths to compare
checkpoint_path_1 = "./lightning_logs/wildfire_progression/mt00kgq8/checkpoints/epoch=58-step=4602.ckpt"
# checkpoint_path_2 = "./lightning_logs/wildfire_progression/f5iz953i/checkpoints/epoch=12-step=1014.ckpt"\
checkpoint_path_2 = "./lightning_logs/wildfire_progression/0dy7djpa/checkpoints/epoch=21-step=1716.ckpt"

checkpoint_path_3 = "/u50/capstone/cs4zp6g17/source/WildfireSpreadPrediction/src/wsts/lightning_logs/wildfire_progression/99p6n9dj/checkpoints/epoch=53-step=4212.ckpt"

# Extract labels from checkpoint paths
checkpoint_label_1 = get_checkpoint_label_from_path(checkpoint_path_1)
checkpoint_label_2 = get_checkpoint_label_from_path(checkpoint_path_2)
checkpoint_label_3 = get_checkpoint_label_from_path(checkpoint_path_3)

data_dir = "/u50/capstone/cs4zp6g17/data/hdf5"
output_dir = "comparison_metrics"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed existing directory: {output_dir}, to ensure fresh output.")
os.makedirs(output_dir, exist_ok=True)

# Run on FULL validation set for comprehensive evaluation
use_full_validation_set = True
num_samples_for_visualization = 10  # Only visualize a few samples
threshold = 0.5

n_leading_observations = 1
n_leading_observations_test_adjustment = 1
crop_side_length = 32
load_from_hdf5 = True
remove_duplicate_features = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def load_model_from_checkpoint(ckpt_path, device):
    """Load model from checkpoint, handling missing config parameters."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = dict(checkpoint['hyper_parameters'])
    else:
        # If hyperparameters not saved, provide defaults
        hparams = {
            'encoder_name': 'resnet18',
            'n_channels': 40,
            'flatten_temporal_dimension': True,
            'pos_class_weight': 236.0,
            'loss_function': 'Focal',
        }
        print(f"  Warning: Using default hyperparameters")
    
    # Ensure all required parameters are present with defaults if missing
    required_params = {
        'encoder_name': 'resnet18',
        'n_channels': 40,
        'flatten_temporal_dimension': True,
        'pos_class_weight': 236.0,
        'loss_function': 'Focal',
    }
    
    for key, default_val in required_params.items():
        if key not in hparams or hparams[key] is None:
            hparams[key] = default_val
            print(f"  Added missing parameter '{key}': {default_val}")
    
    # Remove any extra parameters that SMPModel doesn't accept
    valid_keys = {'encoder_name', 'n_channels', 'flatten_temporal_dimension', 
                  'pos_class_weight', 'loss_function', 'use_doy', 'required_img_size'}
    hparams_filtered = {k: v for k, v in hparams.items() if k in valid_keys}
    
    # Initialize model with filtered hyperparameters
    model = SMPModel(**hparams_filtered)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle wrapped models (e.g., from stage2 training with VQ wrapper)
    # If state_dict has 'seg_model.' prefix, filter it
    if any(k.startswith('seg_model.') for k in state_dict.keys()):
        filtered_state_dict = {k.replace('seg_model.', '', 1): v 
                              for k, v in state_dict.items() 
                              if k.startswith('seg_model.')}
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"  Loaded state dict with 'seg_model.' prefix")
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully")
    return model


# Load all three models
print(f"\nLoading model 1 from {checkpoint_path_1}")
model1 = load_model_from_checkpoint(checkpoint_path_1, device)

print(f"\nLoading model 2 from {checkpoint_path_2}")
model2 = load_model_from_checkpoint(checkpoint_path_2, device)

print(f"\nLoading model 3 (baseline) from {checkpoint_path_3}")
model3 = load_model_from_checkpoint(checkpoint_path_3, device)

print("Initializing data module...")
data_module = FireSpreadDataModule(
    data_dir=data_dir,
    n_leading_observations=n_leading_observations,
    n_leading_observations_test_adjustment=n_leading_observations_test_adjustment,
    crop_side_length=crop_side_length,
    load_from_hdf5=load_from_hdf5,
    remove_duplicate_features=remove_duplicate_features,
    batch_size=1,
    num_workers=0,  # unnecessary overhead for a few samples
    train_val_test_split=[0.8, 0.1, 0.1],
)

print("Setting up validation data for comprehensive evaluation...")
data_module.setup(stage="fit")
val_dataset = data_module.val_dataloader().dataset
print(f"Loaded validation dataset with {len(val_dataset)} samples.")


def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute accuracy, IoU, precision, recall, and F1 for binary prediction."""
    gt_mask = (y_true > threshold).astype(np.uint8)
    pred_mask = (y_pred_prob > threshold).astype(np.uint8)
    
    # Pixel accuracy
    if gt_mask.size != pred_mask.size:
        min_len = min(gt_mask.size, pred_mask.size)
        gt_mask_flat = gt_mask.flatten()[:min_len]
        pred_mask_flat = pred_mask.flatten()[:min_len]
        accuracy = 100.0 * (gt_mask_flat == pred_mask_flat).mean()
    else:
        accuracy = 100.0 * (gt_mask.flatten() == pred_mask.flatten()).mean()
    
    # Region metrics
    tp = int(((gt_mask == 1) & (pred_mask == 1)).sum())
    fp = int(((gt_mask == 0) & (pred_mask == 1)).sum())
    fn = int(((gt_mask == 1) & (pred_mask == 0)).sum())
    tn = int(((gt_mask == 0) & (pred_mask == 0)).sum())
    
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'iou': iou,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def plot_comparison(y_true, y_pred1, y_pred2, y_pred3, idx=None, 
                   label1="Model 1", label2="Model 2", label3="Model 3"):
    """Create a comparison visualization for three model predictions."""
    y_true = y_true.detach().cpu().numpy()
    y_pred1 = torch.sigmoid(y_pred1).detach().cpu().numpy()
    y_pred2 = torch.sigmoid(y_pred2).detach().cpu().numpy()
    y_pred3 = torch.sigmoid(y_pred3).detach().cpu().numpy()

    # Squeeze dimensions if needed
    if y_true.ndim == 3 and y_true.shape[0] == 1:
        y_true = y_true[0]
    if y_pred1.ndim == 3 and y_pred1.shape[0] == 1:
        y_pred1 = y_pred1[0]
    if y_pred2.ndim == 3 and y_pred2.shape[0] == 1:
        y_pred2 = y_pred2[0]
    if y_pred3.ndim == 3 and y_pred3.shape[0] == 1:
        y_pred3 = y_pred3[0]

    # Compute metrics for all three models
    metrics1 = compute_metrics(y_true, y_pred1, threshold)
    metrics2 = compute_metrics(y_true, y_pred2, threshold)
    metrics3 = compute_metrics(y_true, y_pred3, threshold)

    # Create error maps for all three models
    gt_mask = (y_true > threshold).astype(np.uint8)
    pred_mask1 = (y_pred1 > threshold).astype(np.uint8)
    pred_mask2 = (y_pred2 > threshold).astype(np.uint8)
    pred_mask3 = (y_pred3 > threshold).astype(np.uint8)

    h, w = gt_mask.shape
    
    # Error map 1
    error_map1 = np.zeros((h, w, 3), dtype=np.uint8)
    bg = (y_true * 255).astype(np.uint8)
    for c in range(3):
        error_map1[..., c] = bg
    tp1 = (gt_mask == 1) & (pred_mask1 == 1)
    fp1 = (gt_mask == 0) & (pred_mask1 == 1)
    fn1 = (gt_mask == 1) & (pred_mask1 == 0)
    error_map1[tp1] = [0, 200, 0]   # green
    error_map1[fp1] = [200, 0, 0]   # red
    error_map1[fn1] = [0, 0, 200]   # blue

    # Error map 2
    error_map2 = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        error_map2[..., c] = bg
    tp2 = (gt_mask == 1) & (pred_mask2 == 1)
    fp2 = (gt_mask == 0) & (pred_mask2 == 1)
    fn2 = (gt_mask == 1) & (pred_mask2 == 0)
    error_map2[tp2] = [0, 200, 0]   # green
    error_map2[fp2] = [200, 0, 0]   # red
    error_map2[fn2] = [0, 0, 200]   # blue

    # Error map 3
    error_map3 = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        error_map3[..., c] = bg
    tp3 = (gt_mask == 1) & (pred_mask3 == 1)
    fp3 = (gt_mask == 0) & (pred_mask3 == 1)
    fn3 = (gt_mask == 1) & (pred_mask3 == 0)
    error_map3[tp3] = [0, 200, 0]   # green
    error_map3[fp3] = [200, 0, 0]   # red
    error_map3[fn3] = [0, 0, 200]   # blue

    # Probability heatmaps
    prob_cmap = plt.get_cmap("hot")
    prob_img1 = (prob_cmap(y_pred1)[:, :, :3] * 255).astype(np.uint8)
    prob_img2 = (prob_cmap(y_pred2)[:, :, :3] * 255).astype(np.uint8)
    prob_img3 = (prob_cmap(y_pred3)[:, :, :3] * 255).astype(np.uint8)

    # Create comparison figure with 3 rows and 4 columns
    # Row 0: Ground Truth, Model 1 Prediction, Model 2 Prediction, Model 3 Prediction
    # Row 1: Model 1 Error Map, Model 2 Error Map, Model 3 Error Map, Metrics Table
    # Row 2: Model 1 Prob Heatmap, Model 2 Prob Heatmap, Model 3 Prob Heatmap, [empty]
    try:
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25)

        # Row 0: Ground Truth and all Predictions
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pred1 = fig.add_subplot(gs[0, 1])
        ax_pred2 = fig.add_subplot(gs[0, 2])
        ax_pred3 = fig.add_subplot(gs[0, 3])

        ax_gt.imshow(y_true, cmap='gray')
        ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax_gt.axis('off')

        ax_pred1.imshow(pred_mask1, cmap='gray')
        ax_pred1.set_title(f'{label1}\nF1: {metrics1["f1"]:.3f} | Acc: {metrics1["accuracy"]:.2f}%', 
                          fontsize=11, fontweight='bold')
        ax_pred1.axis('off')

        ax_pred2.imshow(pred_mask2, cmap='gray')
        ax_pred2.set_title(f'{label2}\nF1: {metrics2["f1"]:.3f} | Acc: {metrics2["accuracy"]:.2f}%', 
                          fontsize=11, fontweight='bold')
        ax_pred2.axis('off')

        ax_pred3.imshow(pred_mask3, cmap='gray')
        ax_pred3.set_title(f'{label3}\nF1: {metrics3["f1"]:.3f} | Acc: {metrics3["accuracy"]:.2f}%', 
                          fontsize=11, fontweight='bold')
        ax_pred3.axis('off')

        # Row 1: Error Maps and Metrics Table
        ax_err1 = fig.add_subplot(gs[1, 0])
        ax_err2 = fig.add_subplot(gs[1, 1])
        ax_err3 = fig.add_subplot(gs[1, 2])
        ax_metrics = fig.add_subplot(gs[1, 3])

        ax_err1.imshow(error_map1)
        ax_err1.set_title(f'{label1} Error Map\n(G=TP, R=FP, B=FN)', fontsize=10)
        ax_err1.axis('off')

        ax_err2.imshow(error_map2)
        ax_err2.set_title(f'{label2} Error Map\n(G=TP, R=FP, B=FN)', fontsize=10)
        ax_err2.axis('off')

        ax_err3.imshow(error_map3)
        ax_err3.set_title(f'{label3} Error Map\n(G=TP, R=FP, B=FN)', fontsize=10)
        ax_err3.axis('off')

        # Metrics comparison table
        ax_metrics.axis('off')
        
        # Determine best model for each metric
        best_f1 = max(metrics1['f1'], metrics2['f1'], metrics3['f1'])
        best_iou = max(metrics1['iou'], metrics2['iou'], metrics3['iou'])
        best_acc = max(metrics1['accuracy'], metrics2['accuracy'], metrics3['accuracy'])
        
        metrics_text = f"{'Metric':<12} {'M1':<10} {'M2':<10} {'M3':<10}\n"
        metrics_text += "=" * 45 + "\n"
        
        for metric_name in ['f1', 'iou', 'precision', 'recall', 'accuracy']:
            val1 = metrics1[metric_name]
            val2 = metrics2[metric_name]
            val3 = metrics3[metric_name]
            
            if metric_name == 'accuracy':
                best_val = max(val1, val2, val3)
                s1 = f"{val1:.2f}%" + ("*" if val1 == best_val else "")
                s2 = f"{val2:.2f}%" + ("*" if val2 == best_val else "")
                s3 = f"{val3:.2f}%" + ("*" if val3 == best_val else "")
            else:
                best_val = max(val1, val2, val3)
                s1 = f"{val1:.4f}" + ("*" if val1 == best_val else "")
                s2 = f"{val2:.4f}" + ("*" if val2 == best_val else "")
                s3 = f"{val3:.4f}" + ("*" if val3 == best_val else "")
            
            metrics_text += f"{metric_name.upper():<12} {s1:<10} {s2:<10} {s3:<10}\n"
        
        metrics_text += "\n* = best\n"
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=9, verticalalignment='top', family='monospace',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=1'))
        ax_metrics.set_title('Metrics Comparison', fontsize=12, fontweight='bold')

        # Row 2: Probability Heatmaps
        ax_prob1 = fig.add_subplot(gs[2, 0])
        ax_prob2 = fig.add_subplot(gs[2, 1])
        ax_prob3 = fig.add_subplot(gs[2, 2])

        ax_prob1.imshow(prob_img1)
        ax_prob1.set_title(f'{label1} Probability Heatmap', fontsize=10)
        ax_prob1.axis('off')

        ax_prob2.imshow(prob_img2)
        ax_prob2.set_title(f'{label2} Probability Heatmap', fontsize=10)
        ax_prob2.axis('off')

        ax_prob3.imshow(prob_img3)
        ax_prob3.set_title(f'{label3} Probability Heatmap', fontsize=10)
        ax_prob3.axis('off')

        plt.suptitle(f"Three-Model Comparison - Sample {idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"compare_sample_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        print(f"Failed to create comparison figure for sample {idx}: {e}")


# Initialize aggregate metrics for all three models
aggregate_metrics1 = {'accuracy': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
aggregate_metrics2 = {'accuracy': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
aggregate_metrics3 = {'accuracy': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}

# Use full validation set for comprehensive evaluation
if use_full_validation_set:
    indices = list(range(len(val_dataset)))
    print(f"\n{'='*80}")
    print(f"Running FULL VALIDATION on {len(indices)} samples")
    print(f"{'='*80}\n")
    visualize_indices = random.sample(indices, min(num_samples_for_visualization, len(indices)))
else:
    indices = random.sample(range(len(val_dataset)), num_samples_to_show)
    visualize_indices = indices
    print(f"Selected {len(indices)} random samples from validation set.")

for i, idx in enumerate(tqdm(indices, desc="Evaluating models", unit="sample")):
    try:
        x, y = val_dataset[idx]
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)  # add batch dim
        
        with torch.no_grad():
            preds1 = model1(x)
            preds2 = model2(x)
            preds3 = model3(x)
        
        # Only visualize a subset of samples
        if idx in visualize_indices:
            plot_comparison(y.squeeze(0), preds1.squeeze(0), preds2.squeeze(0), preds3.squeeze(0),
                           idx=idx, label1=checkpoint_label_1, label2=checkpoint_label_2, 
                           label3=checkpoint_label_3)
        
        # Collect metrics for aggregate statistics
        y_true_np = y.squeeze(0).detach().cpu().numpy()
        y_pred1_np = torch.sigmoid(preds1.squeeze(0)).detach().cpu().numpy()
        y_pred2_np = torch.sigmoid(preds2.squeeze(0)).detach().cpu().numpy()
        y_pred3_np = torch.sigmoid(preds3.squeeze(0)).detach().cpu().numpy()
        
        if y_true_np.ndim == 3 and y_true_np.shape[0] == 1:
            y_true_np = y_true_np[0]
        if y_pred1_np.ndim == 3 and y_pred1_np.shape[0] == 1:
            y_pred1_np = y_pred1_np[0]
        if y_pred2_np.ndim == 3 and y_pred2_np.shape[0] == 1:
            y_pred2_np = y_pred2_np[0]
        if y_pred3_np.ndim == 3 and y_pred3_np.shape[0] == 1:
            y_pred3_np = y_pred3_np[0]
        
        m1 = compute_metrics(y_true_np, y_pred1_np, threshold)
        m2 = compute_metrics(y_true_np, y_pred2_np, threshold)
        m3 = compute_metrics(y_true_np, y_pred3_np, threshold)
        
        for key in ['accuracy', 'iou', 'precision', 'recall', 'f1']:
            aggregate_metrics1[key].append(m1[key])
            aggregate_metrics2[key].append(m2[key])
            aggregate_metrics3[key].append(m3[key])
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        continue


# Compute and save aggregate results
print("\n" + "="*100)
print("FULL VALIDATION RESULTS - AGGREGATE METRICS ACROSS ALL SAMPLES")
print("="*100)

summary_text = f"\n{'Metric':<15} {checkpoint_label_1:<25} {checkpoint_label_2:<25} {checkpoint_label_3:<25}\n"
summary_text += "="*95 + "\n"

# Track best model overall
best_model_scores = {1: 0, 2: 0, 3: 0}

for metric_name in ['f1', 'iou', 'precision', 'recall', 'accuracy']:
    avg1 = np.mean(aggregate_metrics1[metric_name])
    avg2 = np.mean(aggregate_metrics2[metric_name])
    avg3 = np.mean(aggregate_metrics3[metric_name])
    std1 = np.std(aggregate_metrics1[metric_name])
    std2 = np.std(aggregate_metrics2[metric_name])
    std3 = np.std(aggregate_metrics3[metric_name])
    
    # Find best model for this metric
    best_avg = max(avg1, avg2, avg3)
    if avg1 == best_avg:
        best_model_scores[1] += 1
        marker1, marker2, marker3 = " ⭐", "", ""
    elif avg2 == best_avg:
        best_model_scores[2] += 1
        marker1, marker2, marker3 = "", " ⭐", ""
    else:
        best_model_scores[3] += 1
        marker1, marker2, marker3 = "", "", " ⭐"
    
    if metric_name == 'accuracy':
        summary_text += f"{metric_name.upper():<15} {avg1:>7.2f}% (±{std1:>5.2f}){marker1:<3} {avg2:>7.2f}% (±{std2:>5.2f}){marker2:<3} {avg3:>7.2f}% (±{std3:>5.2f}){marker3:<3}\n"
    else:
        summary_text += f"{metric_name.upper():<15} {avg1:>7.4f} (±{std1:>5.4f}){marker1:<3} {avg2:>7.4f} (±{std2:>5.4f}){marker2:<3} {avg3:>7.4f} (±{std3:>5.4f}){marker3:<3}\n"

# Determine best overall model
best_model_num = max(best_model_scores, key=best_model_scores.get)
best_model_name = [checkpoint_label_1, checkpoint_label_2, checkpoint_label_3][best_model_num - 1]

summary_text += "\n" + "="*95 + "\n"
summary_text += f"BEST MODEL: {best_model_name} (won {best_model_scores[best_model_num]}/5 metrics)\n"
summary_text += "="*95 + "\n"

print(summary_text)

# Save detailed summary to file
summary_path = os.path.join(output_dir, "validation_comparison_summary.txt")
with open(summary_path, 'w') as f:
    f.write("THREE-MODEL CHECKPOINT COMPARISON - FULL VALIDATION RESULTS\n")
    f.write("="*100 + "\n\n")
    f.write(f"Checkpoint 1: {checkpoint_path_1}\n")
    f.write(f"Label: {checkpoint_label_1}\n\n")
    f.write(f"Checkpoint 2: {checkpoint_path_2}\n")
    f.write(f"Label: {checkpoint_label_2}\n\n")
    f.write(f"Checkpoint 3: {checkpoint_path_3}\n")
    f.write(f"Label: {checkpoint_label_3}\n\n")
    f.write(f"Number of samples evaluated: {len(indices)}\n")
    f.write(f"Number of visualizations generated: {len(visualize_indices)}\n")
    f.write(f"Threshold: {threshold}\n\n")
    f.write(summary_text)
    f.write("\n\nDETAILED STATISTICS:\n")
    f.write("="*100 + "\n\n")
    
    for metric_name in ['f1', 'iou', 'precision', 'recall', 'accuracy']:
        f.write(f"\n{metric_name.upper()}:\n")
        f.write(f"  {checkpoint_label_1}: mean={np.mean(aggregate_metrics1[metric_name]):.6f}, std={np.std(aggregate_metrics1[metric_name]):.6f}, min={np.min(aggregate_metrics1[metric_name]):.6f}, max={np.max(aggregate_metrics1[metric_name]):.6f}\n")
        f.write(f"  {checkpoint_label_2}: mean={np.mean(aggregate_metrics2[metric_name]):.6f}, std={np.std(aggregate_metrics2[metric_name]):.6f}, min={np.min(aggregate_metrics2[metric_name]):.6f}, max={np.max(aggregate_metrics2[metric_name]):.6f}\n")
        f.write(f"  {checkpoint_label_3}: mean={np.mean(aggregate_metrics3[metric_name]):.6f}, std={np.std(aggregate_metrics3[metric_name]):.6f}, min={np.min(aggregate_metrics3[metric_name]):.6f}, max={np.max(aggregate_metrics3[metric_name]):.6f}\n")

print(f"\n✅ Detailed summary saved to: {os.path.abspath(summary_path)}")
if visualize_indices:
    print(f"✅ {len(visualize_indices)} sample visualizations saved in: {os.path.abspath(output_dir)}")
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   (Won {best_model_scores[best_model_num]}/5 metrics)")
print("="*100)
