import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import ticker

from models import SMPModel
from dataloader.FireSpreadDataModule import FireSpreadDataModule

checkpoint_path = "lightning_logs/wildfire_progression/99p6n9dj/checkpoints/epoch=53-step=4212.ckpt"
data_dir = "/u50/capstone/cs4zp6g17/data/hdf5"
output_dir = "inference_visualization"
os.makedirs(output_dir, exist_ok=True)

num_samples_to_show = 100
threshold = 0.5

n_leading_observations = 1
n_leading_observations_test_adjustment = 1
crop_side_length = 32
load_from_hdf5 = True
remove_duplicate_features = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Loading model from {checkpoint_path}")

model = SMPModel.load_from_checkpoint(checkpoint_path)
model = model.to(device)
model.eval()

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

print("Setting up test data...")
data_module.setup(stage="test")
test_dataset = data_module.test_dataset
print(f"Loaded test dataset with {len(test_dataset)} samples.")

def plot_fire_prediction(y_true, y_pred, idx=None):
    y_true = y_true.detach().cpu().numpy()
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()

    if y_true.ndim == 3:
        y_true = y_true[0]
    elif y_true.ndim == 1:
        side = int(np.sqrt(y_true.shape[0]))
        y_true = y_true.reshape(side, side)

    if y_pred.ndim == 3:
        y_pred = y_pred[0]
    elif y_pred.ndim == 1:
        side = int(np.sqrt(y_pred.shape[0]))
        y_pred = y_pred.reshape(side, side)

    # If arrays have a leading channel dimension of 1, squeeze it
    if y_true.ndim == 3 and y_true.shape[0] == 1:
        y_true = y_true[0]
    if y_pred.ndim == 3 and y_pred.shape[0] == 1:
        y_pred = y_pred[0]

    # We treat everything as a single timestep here (no GIF/temporal handling)
    def _mask_and_accuracy(gt, pred_prob):
        gt_mask = (gt > threshold).astype(np.uint8)
        pred_mask = (pred_prob > threshold).astype(np.uint8)
        # match sizes
        if gt_mask.size != pred_mask.size:
            min_len = min(gt_mask.size, pred_mask.size)
            gt_mask = gt_mask.flatten()[:min_len]
            pred_mask = pred_mask.flatten()[:min_len]
            acc = 100.0 * (gt_mask == pred_mask).mean()
        else:
            acc = 100.0 * (gt_mask.flatten() == pred_mask.flatten()).mean()
        return gt_mask.reshape(gt.shape), pred_mask.reshape(pred_prob.shape), acc

    # Single-timestep case
    gt_mask, pred_mask, accuracy = _mask_and_accuracy(y_true, y_pred)

    # Compute region-level metrics (IoU, precision, recall, F1)
    tp = int(((gt_mask == 1) & (pred_mask == 1)).sum())
    fp = int(((gt_mask == 0) & (pred_mask == 1)).sum())
    fn = int(((gt_mask == 1) & (pred_mask == 0)).sum())
    tn = int(((gt_mask == 0) & (pred_mask == 0)).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float('nan')
    f1 = float(2 * prec * rec / (prec + rec)) if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else float('nan')


    # Error/difference map: 0=TN,1=FP,2=FN,3=TP -> map to colors
    # We'll create an RGB image where:
    # TP (both 1): green, FP (pred1 gt0): red, FN (pred0 gt1): blue, TN: grayscale background
    h, w = gt_mask.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    # background from ground truth grayscale (0..1 -> 0..255)
    bg = (y_true * 255).astype(np.uint8)
    for c in range(3):
        error_map[..., c] = bg
    tp = (gt_mask == 1) & (pred_mask == 1)
    fp = (gt_mask == 0) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    error_map[tp] = [0, 200, 0]   # green
    error_map[fp] = [200, 0, 0]   # red
    error_map[fn] = [0, 0, 200]   # blue

    # Probability heatmap (0..1) -> use matplotlib colormap
    prob_cmap = plt.get_cmap("hot")
    prob_img = (prob_cmap(y_pred)[:, :, :3] * 255).astype(np.uint8)

    # Create a single combined figure with 7 subplots:
    # Row 0 (5 cols): [0] Ground Truth, [1] Prediction (binary), [2] Overlay, [3] Error map, [4] Prob heatmap
    # Row 1 (5 cols): [0:2] Probability histogram, [2:5] Calibration plot
    try:
        # Single-row layout (top row: 5 panels). Histogram/calibration omitted per request.
        fig = plt.figure(figsize=(22, 6))
        gs = fig.add_gridspec(1, 5, hspace=0.3, wspace=0.2)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[0, 4])

        # Row 0 plots
        ax0.imshow(y_true, cmap='gray')
        ax0.set_title('Ground Truth')
        ax0.axis('off')

        ax1.imshow(pred_mask, cmap='gray')
        ax1.set_title(f'Prediction (Acc: {accuracy:.2f}%)')
        ax1.axis('off')

        ax2.imshow(y_true, cmap='gray')
        ax2.imshow(y_pred, cmap='hot', alpha=0.4, vmin=0, vmax=1)
        ax2.set_title('Overlay (GT + pred prob)')
        # Contours
        try:
            ax2.contour(gt_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=1)
            ax2.contour(pred_mask.astype(float), levels=[0.5], colors=['red'], linewidths=1, linestyles='--')
        except Exception:
            pass
        metrics_txt = f"IoU: {iou:.3f}\nPrec: {prec:.3f}\nRec: {rec:.3f}\nF1: {f1:.3f}"
        ax2.text(0.01, 0.99, metrics_txt, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax2.axis('off')

        ax3.imshow(error_map)
        ax3.set_title('Error map (G=TP, R=FP, B=FN)')
        ax3.axis('off')

        ax4.imshow(prob_img)
        ax4.set_title('Prediction probability heatmap')
        ax4.axis('off')

        # Histogram and calibration plots commented out (user requested).
        # To re-enable, compute `probs = y_pred.flatten()` and `labels = gt_mask.flatten()`
        # then draw the histogram and reliability diagram in a bottom row or separate figure.

        plt.suptitle(f"Sample {idx} — Pixel Acc: {accuracy:.2f}%")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{idx}_combined_acc_{accuracy:.2f}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved combined visualization to {save_path}")
    except Exception as e:
        print(f"Failed to create combined figure: {e}")

# ---- Only pick a few random samples ----
indices = random.sample(range(len(test_dataset)), num_samples_to_show)
print(f"Selected sample indices: {indices}")

for i, idx in enumerate(indices):
    print(f"Processing sample {i+1}/{num_samples_to_show} (index {idx})...")
    x, y = test_dataset[idx]
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        preds = model(x)
    plot_fire_prediction(y.squeeze(0), preds.squeeze(0), idx=idx)

print(f"✅ All predictions saved in: {os.path.abspath(output_dir)}")
