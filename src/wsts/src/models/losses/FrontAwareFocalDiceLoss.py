from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchvision.ops import sigmoid_focal_loss


class FrontAwareFocalDiceLoss(_Loss):
    def __init__(
        self,
        mode: str = "binary",
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        smooth: float = 1.0,
        eps: float = 1e-7,
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        new_fire_weight: float = 0.5,
        front_band_weight: float = 0.2,
        band_width: int = 3,
    ):
        super().__init__()
        if mode != "binary":
            raise ValueError(f"Unsupported mode: {mode}")

        # Torchvision focal loss uses alpha=-1 to disable class balancing.
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            alpha = None

        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.new_fire_weight = new_fire_weight
        self.front_band_weight = front_band_weight
        self.band_width = band_width

    def _front_band(self, prev_fire: torch.Tensor) -> torch.Tensor:
        if prev_fire.ndim == 3:
            prev_fire = prev_fire.unsqueeze(1)

        prev_fire = (prev_fire > 0.5).float()
        kernel_size = 2 * self.band_width + 1
        dilated = F.max_pool2d(
            prev_fire, kernel_size=kernel_size, stride=1, padding=self.band_width
        )
        eroded = 1.0 - F.max_pool2d(
            1.0 - prev_fire,
            kernel_size=kernel_size,
            stride=1,
            padding=self.band_width,
        )
        band = ((dilated - eroded) > 0).float()
        return band.squeeze(1)

    def _weight_map(self, y_true: torch.Tensor, prev_fire: torch.Tensor) -> torch.Tensor:
        y_true = (y_true > 0.5).float()
        prev_fire = (prev_fire > 0.5).float()
        new_fire = ((y_true > 0.5) & (prev_fire < 0.5)).float()
        front_band = self._front_band(prev_fire)
        return 1.0 + self.new_fire_weight * new_fire + self.front_band_weight * front_band

    def _weighted_dice(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(y_pred)
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = (weights * probs * y_true).sum(dim=reduce_dims)
        denominator = (weights * probs).sum(dim=reduce_dims) + (weights * y_true).sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth + self.eps)
        return 1.0 - dice.mean()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        prev_fire: torch.Tensor,
    ) -> torch.Tensor:
        y_true = y_true.float()
        prev_fire = prev_fire.float()
        weights = self._weight_map(y_true, prev_fire)

        focal = sigmoid_focal_loss(
            y_pred,
            y_true,
            alpha=-1 if self.alpha is None else self.alpha,
            gamma=self.gamma,
            reduction="none",
        )
        focal = (focal * weights).sum() / weights.sum().clamp_min(self.eps)
        dice = self._weighted_dice(y_pred, y_true, weights)
        return self.focal_weight * focal + self.dice_weight * dice
