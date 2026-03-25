"""Domain-adversarial UTAE(ResNet18) model for cross-year generalization."""

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only

from .ResNet18UTAELightning import ResNet18UTAELightning
from .transform_blocks import TransformDomainFusionBlock
from .utils import reverse_gradient


class DomainAdversarialUTAELightning(ResNet18UTAELightning):
    """
    Year-adversarial UTAE(Res18) for domain-invariant wildfire prediction.

    Architecture:
    - Shared ResNet18 encoder + LTAE temporal fusion
    - Main head: UNet decoder + segmentation for fire prediction
    - Auxiliary head: optional new-fire delta prediction
    - Auxiliary head: optional year classifier on bottleneck features with gradient reversal

    Training objective:
    - Maximize: fire prediction accuracy (minimize segmentation loss)
    - Emphasize: newly-burned pixels via an auxiliary delta-mask loss
    - Minimize: year prediction accuracy (via gradient reversal)

    This encourages the encoder to learn year-invariant representations while
    paying extra attention to fire-front expansion.
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
        ltae_channels: int = 128,
        d_model: int = 256,
        n_head: int = 16,
        use_doy: bool = False,
        required_img_size: Any = None,
        enable_domain_head: bool = True,
        domain_loss_weight: float = 0.1,
        domain_loss_ramp_epochs: int = 10,
        gradient_reversal_alpha: float = 1.0,
        domain_mlp_hidden: int = 256,
        n_years: int = 8,
        enable_delta_head: bool = False,
        delta_loss_weight: float = 0.2,
        delta_pos_weight_cap: float = 20.0,
        enable_fire_reconstruction: bool = False,
        fire_reconstruction_loss_weight: float = 0.2,
        fire_reconstruction_weight_schedule: str = "constant",
        fire_reconstruction_weight_final: float = 0.0,
        fire_reconstruction_weight_decay_epochs: int = 0,
        fire_mask_min_fraction: float = 0.2,
        fire_mask_max_fraction: float = 0.5,
        fire_mask_strategy: str = "random_rectangle",
        fire_reconstruction_pos_weight: float = 1.0,
        fire_reconstruction_target_timestep: str = "latest",
        fire_reconstruction_history_lookback: int = 3,
        forecast_fusion: str = "early",
        forecast_feature_indices: Optional[List[int]] = None,
        forecast_hidden_dim: int = 128,
        transform_fusion_mode: str = "none",
        transform_fusion_hidden_dim: Optional[int] = None,
        transform_fusion_max_scale: float = 0.1,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            n_channels: Number of input feature channels
            flatten_temporal_dimension: Whether to flatten temporal dim (not used with UTAE)
            pos_class_weight: Weight for positive class in segmentation loss
            loss_function: Segmentation loss function name
            encoder_name: ResNet encoder variant (resnet18, resnet50, etc.)
            encoder_weights: Pretrained weights (imagenet, etc.)
            ltae_channels: Feature channels for LTAE encoder
            d_model: Transformer model dimension
            n_head: Number of attention heads in transformer
            use_doy: Whether to use day-of-year as input
            required_img_size: Required image size for inference crops
            enable_domain_head: Whether to use year adversarial head
            domain_loss_weight: Weight for domain adversarial loss (typical: 0.05-0.2)
            domain_loss_ramp_epochs: Number of epochs to ramp domain loss from 0 to target
            gradient_reversal_alpha: Gradient reversal strength (typical: 0.1-1.0)
            domain_mlp_hidden: Hidden dimension for year classifier MLP
            n_years: Number of year classes (assumes consecutive year indices)
            enable_delta_head: Whether to add an auxiliary head for newly burning pixels
            delta_loss_weight: Weight for the auxiliary delta-mask loss
            delta_pos_weight_cap: Maximum positive class weight used by the delta auxiliary loss
            enable_fire_reconstruction: Whether to add a masked-observation
                reconstruction auxiliary head for the latest observed fire mask.
            fire_reconstruction_loss_weight: Weight for the masked fire
                reconstruction loss.
            fire_reconstruction_weight_schedule: How to schedule the reconstruction
                loss weight over training. "constant" keeps the initial weight, and
                "linear_decay" linearly decays to ``fire_reconstruction_weight_final``.
            fire_reconstruction_weight_final: Final reconstruction weight used when
                ``fire_reconstruction_weight_schedule`` is "linear_decay".
            fire_reconstruction_weight_decay_epochs: Number of epochs over which to
                linearly decay the reconstruction weight. Set to 0 to use the trainer's
                ``max_epochs``.
            fire_mask_min_fraction: Minimum side-length fraction for the random
                masked rectangle applied to the latest observed fire mask.
            fire_mask_max_fraction: Maximum side-length fraction for the random
                masked rectangle applied to the latest observed fire mask.
            fire_mask_strategy: How to place the reconstruction mask. The default
                "random_rectangle" samples anywhere in the crop; "active_fire_rectangle"
                forces overlap with currently burning pixels; "frontier_rectangle"
                focuses the mask on active-fire boundary pixels.
            fire_reconstruction_pos_weight: Positive-class weight used by the
                reconstruction BCE loss on masked pixels.
            fire_reconstruction_target_timestep: Which observed fire timestep to
                reconstruct. "latest" preserves the current behavior; "random_history"
                selects one of the recent historical fire maps instead.
            fire_reconstruction_history_lookback: Number of recent historical
                timesteps eligible when ``fire_reconstruction_target_timestep`` is
                set to ``random_history``.
            forecast_fusion: How to incorporate forecast weather. "early" preserves the
                existing early-fusion path; "film" routes selected forecast channels
                through a separate FiLM-style conditioning branch.
            forecast_feature_indices: Input-channel indices, relative to the final
                dataset tensor, that should be treated as forecast weather features
                when using ``forecast_fusion='film'``.
            forecast_hidden_dim: Hidden size for the forecast FiLM conditioner.
            transform_fusion_mode: Optional transform-domain bottleneck mixing.
                "none" preserves the existing model, while "dct_hadamard"
                applies a lightweight residual transform block to the fused
                temporal bottleneck before decoding.
            transform_fusion_hidden_dim: Hidden size inside the transform-domain
                bottleneck block. Defaults to ``ltae_channels``.
            transform_fusion_max_scale: Maximum residual scaling applied by the
                transform-domain bottleneck.
        """
        forecast_feature_indices = (
            sorted(set(int(idx) for idx in forecast_feature_indices))
            if forecast_feature_indices is not None
            else []
        )
        if forecast_fusion not in {"early", "film"}:
            raise ValueError("forecast_fusion must be either 'early' or 'film'.")
        if transform_fusion_mode not in {"none", "dct_hadamard"}:
            raise ValueError(
                "transform_fusion_mode must be either 'none' or 'dct_hadamard'."
            )

        if forecast_fusion == "film":
            if len(forecast_feature_indices) == 0:
                raise ValueError(
                    "forecast_feature_indices must be provided when forecast_fusion='film'."
                )
            if forecast_feature_indices[0] < 0 or forecast_feature_indices[-1] >= n_channels:
                raise ValueError(
                    "forecast_feature_indices must be valid channel indices for the input tensor."
                )
            encoder_n_channels = n_channels - len(forecast_feature_indices)
            if encoder_n_channels <= 0:
                raise ValueError(
                    "forecast_feature_indices remove all channels from the main encoder path."
                )
        else:
            encoder_n_channels = n_channels

        super().__init__(
            n_channels=encoder_n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            ltae_channels=ltae_channels,
            d_model=d_model,
            n_head=n_head,
            use_doy=use_doy,
            required_img_size=required_img_size,
            *args,
            **kwargs,
        )

        self.enable_domain_head = enable_domain_head
        self.domain_loss_weight = domain_loss_weight
        self.domain_loss_ramp_epochs = domain_loss_ramp_epochs
        self.gradient_reversal_alpha = gradient_reversal_alpha
        self.n_years = n_years
        self.enable_delta_head = enable_delta_head
        self.delta_loss_weight = delta_loss_weight
        self.delta_pos_weight_cap = float(delta_pos_weight_cap)
        self.delta_pos_weight = min(float(pos_class_weight), self.delta_pos_weight_cap)
        self.enable_fire_reconstruction = enable_fire_reconstruction
        self.fire_reconstruction_loss_weight = fire_reconstruction_loss_weight
        self.fire_reconstruction_weight_schedule = fire_reconstruction_weight_schedule
        self.fire_reconstruction_weight_final = float(fire_reconstruction_weight_final)
        self.fire_reconstruction_weight_decay_epochs = int(fire_reconstruction_weight_decay_epochs)
        self.fire_mask_min_fraction = float(fire_mask_min_fraction)
        self.fire_mask_max_fraction = float(fire_mask_max_fraction)
        self.fire_mask_strategy = fire_mask_strategy
        self.fire_reconstruction_pos_weight = float(fire_reconstruction_pos_weight)
        self.fire_reconstruction_target_timestep = fire_reconstruction_target_timestep
        self.fire_reconstruction_history_lookback = int(fire_reconstruction_history_lookback)
        self.total_input_channels = n_channels
        self.forecast_fusion = forecast_fusion
        self.forecast_feature_indices = forecast_feature_indices
        self.forecast_hidden_dim = forecast_hidden_dim
        self.transform_fusion_mode = transform_fusion_mode
        self.transform_fusion_hidden_dim = transform_fusion_hidden_dim
        self.transform_fusion_max_scale = transform_fusion_max_scale

        if not hasattr(self, "hparams"):
            self.hparams = {}
        self.save_hyperparameters(
            "enable_domain_head",
            "domain_loss_weight",
            "domain_loss_ramp_epochs",
            "gradient_reversal_alpha",
            "n_years",
            "enable_delta_head",
            "delta_loss_weight",
            "delta_pos_weight_cap",
            "enable_fire_reconstruction",
            "fire_reconstruction_loss_weight",
            "fire_reconstruction_weight_schedule",
            "fire_reconstruction_weight_final",
            "fire_reconstruction_weight_decay_epochs",
            "fire_mask_min_fraction",
            "fire_mask_max_fraction",
            "fire_mask_strategy",
            "fire_reconstruction_pos_weight",
            "fire_reconstruction_target_timestep",
            "fire_reconstruction_history_lookback",
            "forecast_fusion",
            "forecast_feature_indices",
            "forecast_hidden_dim",
            "transform_fusion_mode",
            "transform_fusion_hidden_dim",
            "transform_fusion_max_scale",
        )

        if self.enable_delta_head:
            self.delta_head = nn.Conv2d(16, 1, kernel_size=1)
        if self.enable_fire_reconstruction:
            if not (0.0 < self.fire_mask_min_fraction <= 1.0):
                raise ValueError("fire_mask_min_fraction must be in (0, 1].")
            if not (0.0 < self.fire_mask_max_fraction <= 1.0):
                raise ValueError("fire_mask_max_fraction must be in (0, 1].")
            if self.fire_mask_min_fraction > self.fire_mask_max_fraction:
                raise ValueError(
                    "fire_mask_min_fraction must be <= fire_mask_max_fraction."
                )
            if self.fire_mask_strategy not in {
                "random_rectangle",
                "active_fire_rectangle",
                "frontier_rectangle",
            }:
                raise ValueError(
                    "fire_mask_strategy must be one of "
                    "{'random_rectangle', 'active_fire_rectangle', 'frontier_rectangle'}."
                )
            if self.fire_reconstruction_pos_weight <= 0.0:
                raise ValueError("fire_reconstruction_pos_weight must be > 0.")
            if self.fire_reconstruction_target_timestep not in {
                "latest",
                "random_history",
            }:
                raise ValueError(
                    "fire_reconstruction_target_timestep must be one of "
                    "{'latest', 'random_history'}."
                )
            if self.fire_reconstruction_history_lookback <= 0:
                raise ValueError("fire_reconstruction_history_lookback must be > 0.")
            if self.fire_reconstruction_weight_schedule not in {"constant", "linear_decay"}:
                raise ValueError("fire_reconstruction_weight_schedule must be 'constant' or 'linear_decay'.")
            if self.fire_reconstruction_weight_decay_epochs < 0:
                raise ValueError("fire_reconstruction_weight_decay_epochs must be >= 0.")
            self.fire_reconstruction_head = nn.Conv2d(16, 1, kernel_size=1)

        if self.forecast_fusion == "film":
            forecast_index_set = set(self.forecast_feature_indices)
            main_feature_indices = [
                idx for idx in range(self.total_input_channels) if idx not in forecast_index_set
            ]
            self.register_buffer(
                "main_feature_indices_tensor",
                torch.tensor(main_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "forecast_feature_indices_tensor",
                torch.tensor(self.forecast_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.forecast_conditioner = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LazyLinear(forecast_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(forecast_hidden_dim, 2 * ltae_channels),
            )

        if self.enable_domain_head:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier = nn.Sequential(
                nn.Linear(ltae_channels, domain_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(domain_mlp_hidden, n_years),
            )
            self.domain_loss_fn = nn.CrossEntropyLoss()

        if self.transform_fusion_mode == "dct_hadamard":
            self.transform_fusion_block = TransformDomainFusionBlock(
                channels=ltae_channels,
                hidden_channels=transform_fusion_hidden_dim,
                max_residual_scale=transform_fusion_max_scale,
            )

    def _extract_prev_fire(self, x: torch.Tensor) -> torch.Tensor:
        """Return the binary active-fire mask from the last observed timestep."""
        return x[:, -1, -1, :, :]

    def _select_reconstruction_timestep(self, seq_len: int, device: torch.device) -> int:
        """Select which observed fire timestep should be reconstructed."""
        if self.fire_reconstruction_target_timestep == "latest" or seq_len <= 1:
            return seq_len - 1

        history_end = seq_len - 1
        history_start = max(0, history_end - self.fire_reconstruction_history_lookback)
        return int(torch.randint(history_start, history_end, (1,), device=device).item())

    def _apply_fire_reconstruction_mask(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask a rectangular region of a selected observed fire mask.

        Returns:
            masked_x: input tensor with the selected binary fire channel partially hidden
            fire_target: original selected binary fire mask
            mask: binary mask where reconstruction loss should be applied
        """
        masked_x = x.clone()
        batch_size, seq_len, _, height, width = x.shape
        target_timestep = self._select_reconstruction_timestep(seq_len, x.device)
        fire_target = x[:, target_timestep, -1, :, :].clone()
        mask = torch.zeros(
            (batch_size, 1, height, width), device=x.device, dtype=x.dtype
        )

        min_h = max(1, int(round(height * self.fire_mask_min_fraction)))
        max_h = max(min_h, int(round(height * self.fire_mask_max_fraction)))
        min_w = max(1, int(round(width * self.fire_mask_min_fraction)))
        max_w = max(min_w, int(round(width * self.fire_mask_max_fraction)))

        active_fire = fire_target > 0.5

        for batch_index in range(batch_size):
            block_h = int(torch.randint(min_h, max_h + 1, (1,), device=x.device).item())
            block_w = int(torch.randint(min_w, max_w + 1, (1,), device=x.device).item())

            candidate_mask: Optional[torch.Tensor]
            if self.fire_mask_strategy == "active_fire_rectangle":
                candidate_mask = active_fire[batch_index]
            elif self.fire_mask_strategy == "frontier_rectangle":
                active_map = active_fire[batch_index].float().unsqueeze(0).unsqueeze(0)
                eroded = -F.max_pool2d(-active_map, kernel_size=3, stride=1, padding=1)
                frontier_mask = (active_map > 0.5) & (eroded < 0.999)
                candidate_mask = frontier_mask.squeeze(0).squeeze(0)
                if not candidate_mask.any():
                    candidate_mask = active_fire[batch_index]
            else:
                candidate_mask = None

            if candidate_mask is not None and candidate_mask.any():
                candidate_coords = candidate_mask.nonzero(as_tuple=False)
                center_index = int(
                    torch.randint(
                        0, candidate_coords.shape[0], (1,), device=x.device
                    ).item()
                )
                center_y, center_x = candidate_coords[center_index].tolist()
                top = max(0, min(int(center_y - block_h // 2), height - block_h))
                left = max(0, min(int(center_x - block_w // 2), width - block_w))
            else:
                top = (
                    int(
                        torch.randint(
                            0, height - block_h + 1, (1,), device=x.device
                        ).item()
                    )
                    if height > block_h
                    else 0
                )
                left = (
                    int(
                        torch.randint(
                            0, width - block_w + 1, (1,), device=x.device
                        ).item()
                    )
                    if width > block_w
                    else 0
                )
            mask[batch_index, 0, top : top + block_h, left : left + block_w] = 1.0

        masked_x[:, target_timestep, -1, :, :] = (
            masked_x[:, target_timestep, -1, :, :] * (1.0 - mask[:, 0, :, :])
        )
        return masked_x, fire_target, mask

    def _get_reconstruction_weight(self) -> float:
        if not self.enable_fire_reconstruction:
            return 0.0
        if self.fire_reconstruction_weight_schedule == "constant":
            return float(self.fire_reconstruction_loss_weight)

        start_weight = float(self.fire_reconstruction_loss_weight)
        final_weight = float(self.fire_reconstruction_weight_final)
        decay_epochs = int(self.fire_reconstruction_weight_decay_epochs)
        if decay_epochs <= 0:
            if self.trainer is not None and self.trainer.max_epochs not in (None, -1):
                decay_epochs = int(self.trainer.max_epochs)
            else:
                decay_epochs = max(int(self.current_epoch) + 1, 1)
        progress = min(max(int(self.current_epoch), 0) / max(decay_epochs, 1), 1.0)
        return start_weight + (final_weight - start_weight) * progress

    def _build_delta_target(self, y: torch.Tensor, prev_fire: torch.Tensor) -> torch.Tensor:
        """Supervise only newly-burning pixels, excluding already active fire."""
        return ((y > 0) & (prev_fire < 0.5)).float()

    def _compute_delta_loss(
        self, delta_logits: torch.Tensor, delta_target: torch.Tensor
    ) -> torch.Tensor:
        """Use a capped BCE objective so the sparse delta target stays numerically stable."""
        pos_weight = torch.tensor(
            self.delta_pos_weight,
            device=delta_logits.device,
            dtype=delta_logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            delta_logits,
            delta_target,
            pos_weight=pos_weight,
        )

    def _compute_fire_reconstruction_loss(
        self,
        reconstruction_logits: torch.Tensor,
        fire_target: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervise only the masked regions to avoid a trivial copy objective.
        """
        per_pixel_loss = F.binary_cross_entropy_with_logits(
            reconstruction_logits,
            fire_target,
            reduction="none",
            pos_weight=torch.tensor(
                self.fire_reconstruction_pos_weight,
                device=reconstruction_logits.device,
                dtype=reconstruction_logits.dtype,
            ),
        )
        masked_loss = per_pixel_loss * reconstruction_mask.squeeze(1)
        normalizer = reconstruction_mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def _forward_outputs(
        self,
        x: torch.Tensor,
        doys: Optional[torch.Tensor] = None,
        years: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Shared forward path that exposes auxiliary outputs for training."""
        del doys

        if self.forecast_fusion == "film":
            forecast_x = x.index_select(2, self.forecast_feature_indices_tensor)
            x = x.index_select(2, self.main_feature_indices_tensor)
        else:
            forecast_x = None

        batch_size, seq_len, channels, height, width = x.shape

        x_flat = x.reshape(batch_size * seq_len, channels, height, width)
        feats_list = self.encoder(x_flat)

        ltae_source = feats_list[self.ltae_feature_index]
        _, ltae_channels, ltae_h, ltae_w = ltae_source.shape
        ltae_temporal = ltae_source.reshape(
            batch_size, seq_len, ltae_channels, ltae_h, ltae_w
        )

        from .utae_paps_models.temporal_fusion import (
            apply_attention_to_scale,
            relative_positions,
        )

        batch_positions = relative_positions(batch_size, seq_len, x.device)
        ltae_fused, attn_mask = self.temporal_encoder(
            ltae_temporal,
            batch_positions=batch_positions,
            pad_mask=None,
        )

        if forecast_x is not None:
            forecast_summary = forecast_x.mean(dim=(-1, -2))
            forecast_params = self.forecast_conditioner(forecast_summary)
            gamma, beta = torch.chunk(forecast_params, 2, dim=1)
            # Bound FiLM modulation so the coarse forecast branch can bias the
            # bottleneck without exploding the decoder or domain head.
            gamma = 0.1 * torch.tanh(gamma)
            beta = 0.1 * torch.tanh(beta)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            ltae_fused = ltae_fused * (1.0 + gamma) + beta

        if self.transform_fusion_mode == "dct_hadamard":
            ltae_fused = self.transform_fusion_block(ltae_fused)

        fused_feats = []
        for feat_index, feat in enumerate(feats_list[1:], start=1):
            if feat_index == self.ltae_feature_index:
                fused_feat = ltae_fused
            else:
                fused_feat = apply_attention_to_scale(
                    feat,
                    attn_mask,
                    batch_size,
                    seq_len,
                )
            fused_feats.append(fused_feat)

        decoded = self.decoder(fused_feats)
        seg_output = self.segmentation_head(decoded)
        delta_output = self.delta_head(decoded) if self.enable_delta_head else None
        reconstruction_output = (
            self.fire_reconstruction_head(decoded)
            if self.enable_fire_reconstruction
            else None
        )

        domain_logits = None
        if self.enable_domain_head and years is not None:
            bottleneck_feat = self.domain_pool(ltae_fused)
            bottleneck_feat = bottleneck_feat.view(batch_size, -1)
            reversed_feat = reverse_gradient(
                bottleneck_feat, self.gradient_reversal_alpha
            )
            domain_logits = self.domain_classifier(reversed_feat)

        return seg_output, delta_output, reconstruction_output, domain_logits

    def forward(
        self,
        x: torch.Tensor,
        doys: Optional[torch.Tensor] = None,
        years: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional year label for adversarial training.

        Args:
            x: Input tensor (B, T, C, H, W)
            doys: Day of year (unused, for compatibility)
            years: Year labels for domain classification (B,).
                   If provided, also returns domain logits.

        Returns:
            If years is None: segmentation logits (B, 1, H, W)
            If years provided: (seg_logits, domain_logits)
        """
        seg_output, _, _, domain_logits = self._forward_outputs(
            x, doys=doys, years=years
        )

        if self.enable_domain_head and years is not None and domain_logits is not None:
            return seg_output, domain_logits

        return seg_output

    def training_step(self, batch, batch_idx):
        """Training step with optional delta-mask and domain-adversarial loss."""
        del batch_idx

        if self.enable_domain_head:
            try:
                if self.hparams.use_doy:
                    x, y, doys, years = batch
                else:
                    x, y, years = batch
                    doys = None
            except ValueError:
                if self.hparams.use_doy:
                    x, y, doys = batch
                else:
                    x, y = batch
                    doys = None
                years = None
        else:
            if self.hparams.use_doy:
                x, y, doys = batch
            else:
                x, y = batch
                doys = None
            years = None

        prev_fire = self._extract_prev_fire(x)
        if self.enable_fire_reconstruction:
            model_x, reconstruction_target, reconstruction_mask = self._apply_fire_reconstruction_mask(x)
        else:
            model_x = x
            reconstruction_target = None
            reconstruction_mask = None

        seg_logits, delta_logits, reconstruction_logits, domain_logits = self._forward_outputs(
            model_x, doys=doys, years=years
        )
        y_hat = seg_logits.squeeze(1)

        seg_loss = self.compute_loss(y_hat, y, prev_fire)
        total_loss = seg_loss

        delta_loss = None
        if self.enable_delta_head and delta_logits is not None:
            delta_target = self._build_delta_target(y, prev_fire)
            delta_loss = self._compute_delta_loss(delta_logits.squeeze(1), delta_target)
            total_loss = total_loss + self.delta_loss_weight * delta_loss

        reconstruction_loss = None
        recon_weight = None
        if (
            self.enable_fire_reconstruction
            and reconstruction_logits is not None
            and reconstruction_target is not None
            and reconstruction_mask is not None
        ):
            reconstruction_loss = self._compute_fire_reconstruction_loss(
                reconstruction_logits.squeeze(1),
                reconstruction_target,
                reconstruction_mask,
            )
            recon_weight = self._get_reconstruction_weight()
            total_loss = total_loss + recon_weight * reconstruction_loss

        if self.enable_domain_head and domain_logits is not None:
            if self.domain_loss_ramp_epochs > 0:
                progress = min(self.current_epoch / self.domain_loss_ramp_epochs, 1.0)
            else:
                progress = 1.0

            effective_domain_weight = self.domain_loss_weight * progress
            years_normalized = years - 2016
            domain_loss = self.domain_loss_fn(domain_logits, years_normalized)
            total_loss = total_loss + effective_domain_weight * domain_loss
        else:
            domain_loss = None
            effective_domain_weight = None

        self.train_f1(y_hat, y)
        self.train_avg_precision(y_hat, y)

        self.log(
            "train_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_AP",
            self.train_avg_precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_seg_loss",
            seg_loss.item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        if delta_loss is not None:
            self.log(
                "train_delta_loss",
                delta_loss.item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        if reconstruction_loss is not None:
            self.log(
                "train_reconstruction_loss",
                reconstruction_loss.item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            if recon_weight is not None:
                self.log(
                    "reconstruction_loss_weight",
                    recon_weight,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )

        if domain_loss is not None:
            self.log(
                "train_domain_loss",
                domain_loss.item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "domain_loss_weight",
                effective_domain_weight,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step using the main segmentation head for AP selection."""
        del batch_idx

        if self.hparams.use_doy:
            x, y, doys = batch[:3]
        else:
            x, y = batch[:2]
            doys = None

        prev_fire = self._extract_prev_fire(x)
        seg_logits, delta_logits, _, _ = self._forward_outputs(
            x, doys=doys, years=None
        )
        y_hat = seg_logits.squeeze(1)

        loss = self.compute_loss(y_hat, y, prev_fire)

        if self.enable_delta_head and delta_logits is not None:
            delta_target = self._build_delta_target(y, prev_fire)
            delta_loss = self._compute_delta_loss(delta_logits.squeeze(1), delta_target)
            self.log(
                "val_delta_loss",
                delta_loss,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        self.val_f1(y_hat, y)
        self.val_avg_precision(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_iou(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "val_f1",
            self.val_f1,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_AP",
            self.val_avg_precision,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_precision",
            self.val_precision,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_recall",
            self.val_recall,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_iou",
            self.val_iou,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
