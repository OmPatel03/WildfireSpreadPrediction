import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import os

from models import SMPModel
from dataloader.FireSpreadDataModule import FireSpreadDataModule

checkpoint_path = "lightning_logs/wildfire_progression/99p6n9dj/checkpoints/epoch=53-step=4212.ckpt"
data_dir = "/u50/capstone/cs4zp6g17/data/hdf5"
output_dir = "predictions_output"
os.makedirs(output_dir, exist_ok=True)

num_samples_to_show = 3
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
    num_workers=4,
    train_val_test_split=[0.8, 0.1, 0.1],
)

print("Setting up test data...")
data_module.setup(stage="test")
test_loader = data_module.test_dataloader()
print("Loaded test dataset.")

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

    print(f"Plotting shapes — y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Ensure both ground truth and prediction are 2D masks
    # If ground truth is continuous, threshold it at the same threshold
    y_true_mask = (y_true > threshold).astype(np.uint8)
    y_pred_mask = (y_pred > threshold).astype(np.uint8)

    # Flatten and compute pixel-wise accuracy
    try:
        accuracy = 100.0 * (y_true_mask.flatten() == y_pred_mask.flatten()).mean()
    except Exception:
        # Fallback in case shapes are weird
        min_len = min(y_true_mask.size, y_pred_mask.size)
        accuracy = 100.0 * (y_true_mask.flatten()[:min_len] == y_pred_mask.flatten()[:min_len]).mean()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(y_true_mask, cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[1].imshow(y_pred_mask, cmap="gray")
    axs[1].set_title(f"Prediction (Acc: {accuracy:.2f}%)")
    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"Sample {idx} — Accuracy: {accuracy:.2f}%")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"sample_{idx}_prediction_acc_{accuracy:.2f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved prediction to {save_path}")

print("Collecting test samples...")
test_batches = list(test_loader)
indices = random.sample(range(len(test_batches)), num_samples_to_show)
print(f"Selected samples: {indices}")

for i, idx in enumerate(indices):
    print(f"Processing sample {i+1}/{num_samples_to_show} (index {idx})...")
    x, y = test_batches[idx]
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        preds = model(x)
    print("Prediction complete. Saving result...")
    plot_fire_prediction(y.squeeze(0), preds.squeeze(0), idx=idx)

print(f"All predictions saved in: {os.path.abspath(output_dir)}")
