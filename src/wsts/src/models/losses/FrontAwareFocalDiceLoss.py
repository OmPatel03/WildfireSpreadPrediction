import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss


class FrontAwareFocalDiceLoss(_Loss):
    def __init__(
        self,
        mode: str = "binary",
        alpha: float | None = None,
        gamma: float = 2.0,
        ignore_index: int | None = None,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        focal_weight: float = 1.0,
        dice_weight: float = 0.3,
        new_fire_weight: float = 0.5,
        front_band_weight: float = 0.2,
        band_width: int = 3,
    ):
        super().__init__()
        if mode != "binary":
            raise ValueError("This loss currently supports only binary mode.")

        self.from_logits = from_logits
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.new_fire_weight = new_fire_weight
        self.front_band_weight = front_band_weight
        self.band_width = band_width

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

    @staticmethod
    def _dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)

    def _front_band(self, prev_fire: torch.Tensor) -> torch.Tensor:
        prev_fire = (prev_fire > 0.5).float()
        outer = self._dilate(prev_fire, kernel_size=2 * self.band_width + 1)
        return (outer - prev_fire).clamp(0.0, 1.0)

    def _bce_map(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        return F.binary_cross_entropy(y_pred, y_true, reduction="none")

    def _masked_mean(self, loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (loss_map * mask).sum() / denom

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        prev_fire: torch.Tensor,
    ) -> torch.Tensor:
        if y_pred.ndim == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)
        if prev_fire.ndim == 3:
            prev_fire = prev_fire.unsqueeze(1)


        y_true = y_true.float()
        prev_fire = prev_fire.float()

        # 1. Global terms
        global_focal = self.focal(y_pred, y_true)
        global_dice = self.dice(y_pred, y_true)

        # 2. Newly burned pixels: y_t \ y_{t-1}
        new_fire_mask = (y_true - prev_fire).clamp(min=0.0, max=1.0)

        # Use BCE-on-logits map as a stable masked focal-like proxy for this term.
        # If you want, this can later be replaced with a custom masked focal map.
        bce_map = self._bce_map(y_pred, y_true)
        new_fire_loss = self._masked_mean(bce_map, new_fire_mask)

        # 3. Front band around previous perimeter
        front_band = self._front_band(prev_fire)
        front_band_loss = self._masked_mean(bce_map, front_band)

        total = (
            self.focal_weight * global_focal
            + self.dice_weight * global_dice
            + self.new_fire_weight * new_fire_loss
            + self.front_band_weight * front_band_loss
        )
        return total