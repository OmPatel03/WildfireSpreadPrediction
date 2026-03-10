import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class FocalTverskyLoss(_Loss):
    """
    Focal Tversky Loss for binary segmentation.
    
    Combines the Tversky index with focal weighting to handle class imbalance.
    
    Tversky Index: TI = TP / (TP + α·FP + β·FN)
    Focal Tversky Loss: FTL = (1 - TI)^γ
    
    Args:
        alpha: Weight for false positives. Default 0.5 (symmetric).
               Reduce for class imbalance (e.g., 0.3 to penalize FP less).
        beta: Weight for false negatives. Default 0.5 (symmetric).
              Increase for class imbalance (e.g., 0.7 to penalize FN more).
        gamma: Focusing parameter. Default 1.0. Increase (e.g., 2.0) for harder focusing.
        from_logits: If True, applies sigmoid to predictions. Default True.
        smooth: Smoothing constant to avoid division by zero. Default 1e-7.
        ignore_index: Index to ignore in loss computation. Default None.
    
    References:
        Abraham, N., Khan, M. S. (2019). A Novel Focal Tversky Loss Function with 
        Improved Attention U-Net for Segmentation.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        from_logits: bool = True,
        smooth: float = 1e-7,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Tversky Loss.
        
        Args:
            y_pred: Predicted logits or probabilities, shape (B, C, H, W) or (B, H, W).
            y_true: Ground truth binary labels, shape (B, C, H, W) or (B, H, W).
        
        Returns:
            Scalar loss value.
        """
        # Ensure 4D shape (B, C, H, W)
        if y_pred.ndim == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)
        
        # Convert to probabilities if needed
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.float()
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B*C, H*W)
        B, C, H, W = y_pred.shape
        y_pred_flat = y_pred.reshape(B * C, -1)
        y_true_flat = y_true.reshape(B * C, -1)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = y_true_flat != self.ignore_index
            y_pred_flat = y_pred_flat * mask.float()
            y_true_flat = y_true_flat * mask.float()
        
        # Compute TP, FP, FN
        tp = (y_pred_flat * y_true_flat).sum(dim=1)
        fp = (y_pred_flat * (1.0 - y_true_flat)).sum(dim=1)
        fn = ((1.0 - y_pred_flat) * y_true_flat).sum(dim=1)
        
        # Tversky Index
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal Tversky Loss
        focal_tversky = 1.0 - tversky_index
        focal_tversky_loss = torch.pow(focal_tversky, self.gamma)
        
        # Return mean across batch
        return focal_tversky_loss.mean()
