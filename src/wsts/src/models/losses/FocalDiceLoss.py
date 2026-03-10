import torch
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

class FocalDiceLoss(_Loss):
    def __init__(
        self,
        mode: str = "binary",
        alpha: float | None = None,
        gamma: float = 2.0,
        ignore_index: int | None = None,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        if mode not in {"binary", "multiclass", "multilabel"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal = FocalLoss(
            mode=mode,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(
            mode=mode,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        focal = self.focal(y_pred, y_true)
        dice = self.dice(y_pred, y_true)
        return self.focal_weight * focal + self.dice_weight * dice