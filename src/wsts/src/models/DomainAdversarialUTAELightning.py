"""Domain-adversarial UTAE(ResNet18) model for cross-year generalization."""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from .ResNet18UTAELightning import ResNet18UTAELightning
from .utils import reverse_gradient


class DomainAdversarialUTAELightning(ResNet18UTAELightning):
    """
    Year-adversarial UTAE(Res18) for domain-invariant wildfire prediction.
    
    Architecture:
    - Shared ResNet18 encoder + LTAE temporal fusion
    - Main head: UNet decoder + segmentation for fire prediction
    - Auxiliary head: Year classifier on bottleneck features with gradient reversal
    
    Training objective:
    - Maximize: fire prediction accuracy (minimize segmentation loss)
    - Minimize: year prediction accuracy (via gradient reversal)
    
    This encourages the encoder to learn year-invariant representations.
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
        n_years: int = 8,  # 2016-2023 = 8 years
        *args: Any, 
        **kwargs: Any
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
            domain_loss_ramp_epochs: Number of epochs to ramp domain loss from 0 to target (default: 0, i.e., start immediately)
            gradient_reversal_alpha: Gradient reversal strength (typical: 0.1-1.0)
            domain_mlp_hidden: Hidden dimension for year classifier MLP
            n_years: Number of year classes (assumes consecutive year indices)
        """
        super().__init__(
            n_channels=n_channels,
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
            **kwargs
        )
        
        self.enable_domain_head = enable_domain_head
        self.domain_loss_weight = domain_loss_weight
        self.domain_loss_ramp_epochs = domain_loss_ramp_epochs
        self.gradient_reversal_alpha = gradient_reversal_alpha
        self.n_years = n_years
        
        # Save hyperparams for access in training_step
        if not hasattr(self, 'hparams'):
            self.hparams = {}
        self.save_hyperparameters(
            'enable_domain_head', 'domain_loss_weight', 'domain_loss_ramp_epochs',
            'gradient_reversal_alpha', 'n_years'
        )
        
        if self.enable_domain_head:
            # Global average pool bottleneck features
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            
            # Year classifier MLP
            # Input: ltae_channels (pooled from bottleneck)
            self.domain_classifier = nn.Sequential(
                nn.Linear(ltae_channels, domain_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(domain_mlp_hidden, n_years),
            )
            
            # CrossEntropyLoss for year classification
            self.domain_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        x: torch.Tensor,
        doys: Optional[torch.Tensor] = None,
        years: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
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
        batch_size, seq_len, channels, height, width = x.shape
        
        # Encoder forward
        x_flat = x.reshape(batch_size * seq_len, channels, height, width)
        feats_list = self.encoder(x_flat)
        
        # LTAE temporal fusion
        ltae_source = feats_list[self.ltae_feature_index]
        _, ltae_channels, ltae_h, ltae_w = ltae_source.shape
        ltae_temporal = ltae_source.reshape(
            batch_size, seq_len, ltae_channels, ltae_h, ltae_w
        )
        
        from .utae_paps_models.temporal_fusion import relative_positions, apply_attention_to_scale
        batch_positions = relative_positions(batch_size, seq_len, x.device)
        
        ltae_fused, attn_mask = self.temporal_encoder(
            ltae_temporal,
            batch_positions=batch_positions,
            pad_mask=None,
        )
        
        # Apply attention to all scales
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
        
        # Decoder + segmentation head
        decoded = self.decoder(fused_feats)
        seg_output = self.segmentation_head(decoded)
        
        # Domain classifier (if enabled and years provided)
        if self.enable_domain_head and years is not None:
            # Get bottleneck features from LTAE output
            # ltae_fused shape: (B, C, H, W)
            bottleneck_feat = self.domain_pool(ltae_fused)  # (B, C, 1, 1)
            bottleneck_feat = bottleneck_feat.view(batch_size, -1)  # (B, C)
            
            # Apply gradient reversal
            reversed_feat = reverse_gradient(bottleneck_feat, self.gradient_reversal_alpha)
            
            # Classify year
            domain_logits = self.domain_classifier(reversed_feat)  # (B, n_years)
            
            return seg_output, domain_logits
        
        return seg_output
    
    def training_step(self, batch, batch_idx):
        """
        Training step with domain adversarial loss.
        
        Batch format:
        - Standard: (x, y, prev_fire)
        - With years: (x, y, prev_fire, years) [depends on dataset return_year setting]
        """
        # Unpack batch - handle optional years
        if self.enable_domain_head:
            try:
                if self.hparams.use_doy:
                    # (x, y, doys, years)
                    x, y, doys, years = batch
                else:
                    # (x, y, years)
                    x, y, years = batch
            except ValueError:
                # Fallback if years not in batch
                if self.hparams.use_doy:
                    x, y, doys = batch
                else:
                    x, y = batch
                years = None
        else:
            if self.hparams.use_doy:
                x, y, doys = batch
            else:
                x, y = batch
            years = None
        
        # Extract prev_fire from input (previous fire mask is typically channel 0 of last timestep)
        # But for safety, use channel 1 which is usually vegetation/fire mask
        prev_fire = x[:, -1, 1, :, :] if x.shape[2] > 1 else x[:, -1, 0, :, :]
        
        # Forward pass
        if self.enable_domain_head and years is not None:
            y_hat, domain_logits = self(x, doys=None, years=years)
        else:
            y_hat = self(x)
            domain_logits = None
        y_hat = y_hat.squeeze(1)
        
        # Compute segmentation loss
        seg_loss = self.compute_loss(y_hat, y, prev_fire)
        
        # Compute domain loss (if enabled)
        if self.enable_domain_head and domain_logits is not None:
            # Ramp up domain loss weight over first N epochs
            if self.domain_loss_ramp_epochs > 0:
                progress = min(self.current_epoch / self.domain_loss_ramp_epochs, 1.0)
            else:
                progress = 1.0
            
            effective_domain_weight = self.domain_loss_weight * progress
            
            # Normalize years to 0-indexed for year classifier (if needed)
            # Assuming years are in range [2016, 2023], convert to [0, 7]
            years_normalized = years - 2016
            domain_loss = self.domain_loss_fn(domain_logits, years_normalized)
            
            total_loss = seg_loss + effective_domain_weight * domain_loss
        else:
            domain_loss = None
            total_loss = seg_loss
        
        # Compute metrics
        self.train_f1(y_hat, y)
        self.train_avg_precision(y_hat, y)
        
        # Logging
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
        
        if domain_loss is not None:
            self.log(
                "train_seg_loss",
                seg_loss.item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
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
        """Validation step (standard, no domain loss)."""
        # Unpack batch - handle optional years
        if self.hparams.use_doy:
            # Could be (x, y, doys) or (x, y, doys, years)
            x, y, doys = batch[:3]
        else:
            # Could be (x, y) or (x, y, years)
            x, y = batch[:2]
        
        prev_fire = x[:, -1, 1, :, :] if x.shape[2] > 1 else x[:, -1, 0, :, :]
        
        # Forward pass (segmentation only)
        y_hat = self(x).squeeze(1)
        
        # Compute loss and metrics
        loss = self.compute_loss(y_hat, y, prev_fire)
        
        # Update metrics first, then log them separately (consistent with training_step)
        self.val_avg_precision(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "val_AP",
            self.val_avg_precision,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
