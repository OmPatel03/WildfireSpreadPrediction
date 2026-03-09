import argparse

# --- Patch argparse for Python 3.12 + LightningCLI compatibility ---
# Force _parse_known_args to always accept 'intermixed' keyword even if older call sites omit it
if "intermixed" in argparse.ArgumentParser._parse_known_args.__code__.co_varnames:
    # Already has the new signature (Python 3.12) — wrap for backward compatibility
    _orig_parse_known_args = argparse.ArgumentParser._parse_known_args

    def _parse_known_args_fixed(self, arg_strings, namespace, *args, **kwargs):
        # Always provide default intermixed=False if caller doesn't specify it
        if "intermixed" not in kwargs:
            kwargs["intermixed"] = False
        return _orig_parse_known_args(self, arg_strings, namespace, **kwargs)

    argparse.ArgumentParser._parse_known_args = _parse_known_args_fixed
# -------------------------------------------------------------------

from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
import wandb

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values
from models import BaseModel
from models.vq_priority_wrapper import VQPriorityWrapper
from models.vq_vae import VQVAE, VQOutput


class VQPriorityStage1Module(pl.LightningModule):
    """Joint training for segmentation + VQ-VAE context encoder."""

    def __init__(
        self,
        seg_model: BaseModel,
        vqvae: VQVAE,
        vq_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["seg_model", "vqvae"])

        self.seg_model = seg_model
        self.vqvae = vqvae

        self.vq_wrapper = VQPriorityWrapper(
            segmentation_model=self.seg_model,
            vqvae=self.vqvae,
            feature_extractor=self._default_feature_extractor,
        )

        self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        
        # Fire-specific metrics
        self.val_fire_precision = torchmetrics.Precision("binary")
        self.val_fire_recall = torchmetrics.Recall("binary")
        self.val_fire_f1 = torchmetrics.F1Score("binary")

    def _prepare_seg_input(self, x: torch.Tensor) -> torch.Tensor:
        if (
            hasattr(self.seg_model, "hparams")
            and getattr(self.seg_model.hparams, "flatten_temporal_dimension", False)
            and x.dim() == 5
        ):
            return x.flatten(start_dim=1, end_dim=2)
        return x

    def _default_feature_extractor(
        self,
        seg_model: BaseModel,
        x: torch.Tensor,
        doys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(seg_model, "model") and hasattr(seg_model.model, "encoder"):
            features = seg_model.model.encoder(x)
            if isinstance(features, (list, tuple)):
                return features[-1]
            return features
        raise ValueError(
            "Segmentation model does not expose an encoder. Provide a feature_extractor."
        )

    def _unpack_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if hasattr(self.seg_model, "hparams") and getattr(self.seg_model.hparams, "use_doy", False):
            x, y, doys = batch
            return x, y, doys
        x, y = batch
        return x, y, None

    def forward(self, x: torch.Tensor, doys: Optional[torch.Tensor] = None):
        return self.seg_model(x, doys) if doys is not None else self.seg_model(x)

    def training_step(self, batch, batch_idx):
        x, y, doys = self._unpack_batch(batch)

        y_hat, y = self.seg_model.get_pred_and_gt(batch)
        seg_loss = self.seg_model.compute_loss(y_hat, y)

        seg_input = self._prepare_seg_input(x)
        bottleneck = self.vq_wrapper._extract_bottleneck(seg_input, doys)
        vq_out: VQOutput = self.vqvae(bottleneck)

        total_vq_loss = (
            self.hparams.vq_loss_weight * vq_out.vq_loss
            + self.hparams.recon_loss_weight * vq_out.recon_loss
        )

        loss = seg_loss + total_vq_loss

        f1 = self.train_f1(y_hat, y)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_seg_loss", seg_loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_vq_loss", vq_out.vq_loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_recon_loss", vq_out.recon_loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, doys = self._unpack_batch(batch)

        y_hat, y = self.seg_model.get_pred_and_gt(batch)
        seg_loss = self.seg_model.compute_loss(y_hat, y)

        seg_input = self._prepare_seg_input(x)
        bottleneck = self.vq_wrapper._extract_bottleneck(seg_input, doys)
        vq_out: VQOutput = self.vqvae(bottleneck)

        total_vq_loss = (
            self.hparams.vq_loss_weight * vq_out.vq_loss
            + self.hparams.recon_loss_weight * vq_out.recon_loss
        )

        loss = seg_loss + total_vq_loss

        f1 = self.val_f1(y_hat, y)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_seg_loss", seg_loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_vq_loss", vq_out.vq_loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_recon_loss", vq_out.recon_loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log fire-specific metrics
        y_probs = y_hat if y_hat.dim() == y.dim() else torch.sigmoid(y_hat)
        fire_prec = self.val_fire_precision(y_probs, y)
        fire_rec = self.val_fire_recall(y_probs, y)
        fire_f1 = self.val_fire_f1(y_probs, y)
        self.log("val_fire_precision", fire_prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_fire_recall", fire_rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_fire_f1", fire_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.fail_untyped = False

        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        parser.add_argument("--do_train", type=bool, help="If True: run training.")
        parser.add_argument("--do_validate", type=bool, default=False, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming.")

    def before_instantiate_classes(self):
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features,
        )

        self.config.model.seg_model.init_args.n_channels = n_features

        train_years, _, _ = FireSpreadDataModule.split_fires(self.config.data.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)
        self.config.model.seg_model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        if wandb.run is None:
            wandb.init(project="WildfireSpreadPrediction", name="vq_priority_stage1")
        config_file_name = f"{wandb.run.dir}/cli_config.yaml"

        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")
        
        # Fire-specific metrics
        wandb.define_metric("val_fire_precision", summary="max")
        wandb.define_metric("val_fire_recall", summary="max")
        wandb.define_metric("val_fire_f1", summary="max")


def main():
    cli = MyLightningCLI(
        VQPriorityStage1Module,
        FireSpreadDataModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False,
    )
    cli.wandb_setup()

    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt_path)

    if cli.config.do_validate:
        ckpt = cli.config.ckpt_path
        if cli.config.do_train:
            ckpt = "best"
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    main()