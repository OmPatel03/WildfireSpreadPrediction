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

from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values
from models import BaseModel
from models.vq_priority_wrapper import VQPriorityWrapper
from models.vq_vae import VQVAE
from utils.priority_sampler import (
    build_weighted_sampler,
    compute_cluster_ids,
    compute_sample_weights,
)


def _load_prefixed_state_dict(
    target: torch.nn.Module, state_dict: dict, prefix: str
) -> None:
    filtered = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not filtered:
        raise ValueError(f"No keys with prefix '{prefix}' found in checkpoint.")
    target.load_state_dict(filtered, strict=True)


def _default_feature_extractor(
    seg_model: BaseModel, x: torch.Tensor, doys: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if hasattr(seg_model, "model") and hasattr(seg_model.model, "encoder"):
        features = seg_model.model.encoder(x)
        if isinstance(features, (list, tuple)):
            return features[-1]
        return features

    if hasattr(seg_model, "model") and seg_model.model.__class__.__name__ == "UTAE":
        seg_model.model.encoder = True
        seg_model.model.return_maps = True
        bottleneck, _ = seg_model.model(x, batch_positions=doys, return_att=False)
        seg_model.model.encoder = False
        seg_model.model.return_maps = False
        return bottleneck

    raise ValueError(
        "Segmentation model does not expose an encoder. Provide a feature_extractor."
    )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.fail_untyped = False

        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path", "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool, help="If True: run training.")
        parser.add_argument("--do_validate", type=bool, default=True, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming.")
        parser.add_argument(
            "--stage1_ckpt_path",
            type=str,
            required=True,
            help="Checkpoint from stage 1 (VQPriorityStage1Module).",
        )
        parser.add_class_arguments(VQVAE, "vqvae")

    def before_instantiate_classes(self):
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features,
        )

        self.config.model.init_args.n_channels = n_features

        train_years, _, _ = FireSpreadDataModule.split_fires(self.config.data.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)
        self.config.model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        self.wandb_setup()

        checkpoint = torch.load(self.config.stage1_ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        _load_prefixed_state_dict(self.model, state_dict, "seg_model.")

        vqvae_cfg = dict(self.config.vqvae)
        vqvae = VQVAE(**vqvae_cfg)
        _load_prefixed_state_dict(vqvae, state_dict, "vqvae.")
        vqvae.eval()
        for param in vqvae.parameters():
            param.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vqvae.to(device)
        self.model.to(device)

        self.datamodule.setup("fit")

        # Strategy: Subtle VQ Weighting with Fire-Density (Experiment 3 / Option C)
        # Blends VQ clustering dominantly with fire-density in a gentle way
        # final_weight = vq_weight^0.9 * fire_density^0.1
        # This keeps VQ as the primary signal but adds a mild fire-density nudge
        print("="*80)
        print("EXPERIMENT: Subtle VQ + Fire-Density Weighting (Option C)")
        print("="*80)
        
        # Step 1: Compute VQ clusters
        print("Step 1: Computing VQ cluster IDs...")
        vq_wrapper = VQPriorityWrapper(
            segmentation_model=self.model,
            vqvae=vqvae,
            feature_extractor=_default_feature_extractor,
        )

        train_loader = DataLoader(
            self.datamodule.train_dataset,
            batch_size=self.datamodule.batch_size,
            shuffle=False,
            num_workers=self.datamodule.num_workers,
            pin_memory=True,
        )

        cluster_ids = compute_cluster_ids(
            vq_wrapper=vq_wrapper,
            dataloader=train_loader,
            device=device,
            flatten_temporal=self.model.hparams.flatten_temporal_dimension,
        )
        
        # Step 2: Compute fire density for each sample
        print("Step 2: Computing fire density for each sample...")
        fire_densities = []
        for i in range(len(self.datamodule.train_dataset)):
            try:
                _, y = self.datamodule.train_dataset[i]
                fire_rate = y.float().mean().item()
                fire_densities.append(fire_rate)
            except Exception as e:
                print(f"Warning: Could not compute fire density for sample {i}: {e}")
                fire_densities.append(0.5)  # Use median as fallback
        
        fire_densities = torch.tensor(fire_densities, dtype=torch.float32)
        # Normalize fire densities to [0.5, 1.5] range (gentle modifier)
        fire_densities_min = fire_densities.min()
        fire_densities_max = fire_densities.max()
        if fire_densities_max > fire_densities_min:
            fire_densities_norm = 0.5 + (fire_densities - fire_densities_min) / (fire_densities_max - fire_densities_min)
        else:
            fire_densities_norm = torch.ones_like(fire_densities)
        
        # Step 3: Blend VQ weights with fire-density (gently)
        print("Step 3: Blending VQ and fire-density weights (90% VQ, 10% fire-density)...")
        vq_weights = compute_sample_weights(cluster_ids, vqvae.codebook.num_embeddings)
        
        # Gentle blending: weight = vq_weight^0.9 * fire_density^0.1
        # This keeps VQ dominant but adds a mild signal from fire density
        blended_weights = (vq_weights ** 0.9) * (fire_densities_norm ** 0.1)
        blended_weights = blended_weights / blended_weights.mean()  # Normalize
        
        sampler = build_weighted_sampler(blended_weights)

        def _weighted_train_loader():
            return DataLoader(
                self.datamodule.train_dataset,
                batch_size=self.datamodule.batch_size,
                sampler=sampler,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )

        self.datamodule.train_dataloader = _weighted_train_loader
        
        print(f"Blended weights computed.")
        print(f"  VQ weights: mean={vq_weights.mean():.3f}, max={vq_weights.max():.3f}, min={vq_weights.min():.3f}")
        print(f"  Fire density (normalized): mean={fire_densities_norm.mean():.3f}, max={fire_densities_norm.max():.3f}, min={fire_densities_norm.min():.3f}")
        print(f"  Blended weights: mean={blended_weights.mean():.3f}, max={blended_weights.max():.3f}, min={blended_weights.min():.3f}")
        print("="*80)

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        if wandb.run is None:
            wandb.init(project="WildfireSpreadPrediction", name="stage2_vq_weighted_optionc")
        config_file_name = f"{wandb.run.dir}/cli_config.yaml"

        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        
        # Define metrics for WandB
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
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
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
