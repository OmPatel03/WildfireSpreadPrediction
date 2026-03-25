from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional, Union
from .cluster_sampling import load_cluster_assignments, build_sample_weights


class FireSpreadDataModule(LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, n_leading_observations_test_adjustment: int,
                 crop_side_length: int,
                 load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool,
                 features_to_keep: Union[Optional[List[int]], str] = None, return_doy: bool = False, return_year: bool = False,
                 degree_encoding: str = "sin",
                 data_fold_id: int = 0,
                 enable_cluster_sampling: bool = False,
                 cluster_assignments_path: Optional[str] = None,
                 cluster_sampling_alpha: float = 1.0,
                 cluster_sampling_min_count: int = 1,
                 cluster_sampling_mix_uniform: float = 0.2,
                 cluster_sampling_max_weight: float = 0.0,
                 cluster_sampling_replacement: bool = True,
                 cluster_sampling_log_stats: bool = True,
                 frontier_sampling_enabled: bool = False,
                 frontier_band_width: int = 16,
                 frontier_boundary_ratio: float = 0.5,
                 frontier_interior_ratio: float = 0.25,
                 frontier_hard_negative_ratio: float = 0.25,
                 frontier_center_jitter: int = 8,
                 crop_search_trials: int = 10,
                 *args, **kwargs):
        """_summary_ Data module for loading the WildfireSpreadTS dataset.

        Args:
            data_dir (str): _description_ Path to the directory containing the data.
            batch_size (int): _description_ Batch size for training and validation set. Test set uses batch size 1, because images of different sizes can not be batched together.
            n_leading_observations (int): _description_ Number of days to use as input observation.
            n_leading_observations_test_adjustment (int): _description_ When increasing the number of leading observations, the number of samples per fire is reduced.
              This parameter allows to adjust the number of samples in the test set to be the same across several different values of n_leading_observations,
              by skipping some initial fires. For example, if this is set to 5, and n_leading_observations is set to 1, the first four samples that would be
              in the test set are skipped. This way, the test set is the same as it would be for n_leading_observations=5, thereby retaining comparability
              of the test set.
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF.
            num_workers (int): _description_ Number of workers for the dataloader.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            features_to_keep (Union[Optional[List[int]], str], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            degree_encoding (str, optional): Angular-feature encoding used by the dataset preprocessing.
            data_fold_id (int, optional): _description_. Which data fold to use, i.e. splitting years into train/val/test set. Defaults to 0.
            frontier_sampling_enabled (bool, optional): Enable frontier-centered crop selection during training.
            frontier_band_width (int, optional): Radius in pixels used to define the frontier band and near-front negative zone.
            frontier_boundary_ratio (float, optional): Relative sampling weight for boundary-centered crops.
            frontier_interior_ratio (float, optional): Relative sampling weight for interior-fire crops.
            frontier_hard_negative_ratio (float, optional): Relative sampling weight for near-front hard-negative crops.
            frontier_center_jitter (int, optional): Random center jitter applied to frontier-centered crops.
            crop_search_trials (int, optional): Number of candidate crops evaluated by the baseline fire-preferring crop search.
        """
        super().__init__()

        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.data_fold_id = data_fold_id
        self.enable_cluster_sampling = enable_cluster_sampling
        self.cluster_assignments_path = cluster_assignments_path
        self.cluster_sampling_alpha = cluster_sampling_alpha
        self.cluster_sampling_min_count = cluster_sampling_min_count
        self.cluster_sampling_mix_uniform = cluster_sampling_mix_uniform
        self.cluster_sampling_max_weight = cluster_sampling_max_weight
        self.cluster_sampling_replacement = cluster_sampling_replacement
        self.cluster_sampling_log_stats = cluster_sampling_log_stats
        self.frontier_sampling_enabled = frontier_sampling_enabled
        self.frontier_band_width = frontier_band_width
        self.frontier_boundary_ratio = frontier_boundary_ratio
        self.frontier_interior_ratio = frontier_interior_ratio
        self.frontier_hard_negative_ratio = frontier_hard_negative_ratio
        self.frontier_center_jitter = frontier_center_jitter
        self.crop_search_trials = crop_search_trials
        self.return_doy = return_doy
        self.return_year = return_year
        self.degree_encoding = degree_encoding
        # wandb apparently can't pass None values via the command line without turning them into a string, so we need this workaround
        self.features_to_keep = features_to_keep if type(
            features_to_keep) != str else None
        self.remove_duplicate_features = remove_duplicate_features
        self.num_workers = num_workers
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.train_sampler = None

    def setup(self, stage: str):
        train_years, val_years, test_years = self.split_fires(
            self.data_fold_id)
        self.train_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=train_years,
                                               n_leading_observations=self.n_leading_observations,
                                               n_leading_observations_test_adjustment=None,
                                               crop_side_length=self.crop_side_length,
                                               load_from_hdf5=self.load_from_hdf5, is_train=True,
                                               remove_duplicate_features=self.remove_duplicate_features,
                                               features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                               return_year=self.return_year, degree_encoding=self.degree_encoding, stats_years=train_years,
                                               frontier_sampling_enabled=self.frontier_sampling_enabled,
                                               frontier_band_width=self.frontier_band_width,
                                               frontier_boundary_ratio=self.frontier_boundary_ratio,
                                               frontier_interior_ratio=self.frontier_interior_ratio,
                                               frontier_hard_negative_ratio=self.frontier_hard_negative_ratio,
                                               frontier_center_jitter=self.frontier_center_jitter,
                                               crop_search_trials=self.crop_search_trials)
        self.val_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=val_years,
                                             n_leading_observations=self.n_leading_observations,
                                             n_leading_observations_test_adjustment=None,
                                             crop_side_length=self.crop_side_length,
                                             load_from_hdf5=self.load_from_hdf5, is_train=True,
                                             remove_duplicate_features=self.remove_duplicate_features,
                                             features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                             return_year=self.return_year, degree_encoding=self.degree_encoding, stats_years=train_years,
                                             frontier_sampling_enabled=False,
                                             crop_search_trials=self.crop_search_trials)
        self.test_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=test_years,
                                              n_leading_observations=self.n_leading_observations,
                                              n_leading_observations_test_adjustment=self.n_leading_observations_test_adjustment,
                                              crop_side_length=self.crop_side_length,
                                              load_from_hdf5=self.load_from_hdf5, is_train=False,
                                              remove_duplicate_features=self.remove_duplicate_features,
                                              features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                              return_year=self.return_year, degree_encoding=self.degree_encoding, stats_years=train_years,
                                              frontier_sampling_enabled=False,
                                              crop_search_trials=self.crop_search_trials)
        self.train_sampler = self._build_train_sampler()

    def train_dataloader(self):
        if self.train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def _build_train_sampler(self):
        if not self.enable_cluster_sampling:
            return None
        if self.cluster_assignments_path is None:
            raise ValueError("enable_cluster_sampling=True requires cluster_assignments_path.")

        if self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1:
            raise RuntimeError(
                "Cluster sampling currently supports single-process training only. "
                "Set trainer.devices=1 (or disable cluster sampling)."
            )

        assignment_map = load_cluster_assignments(self.cluster_assignments_path)
        sample_keys = self.train_dataset.get_all_sample_keys()

        sample_weights, stats = build_sample_weights(
            sample_keys=sample_keys,
            assignment_map=assignment_map,
            alpha=self.cluster_sampling_alpha,
            min_count=self.cluster_sampling_min_count,
            mix_uniform=self.cluster_sampling_mix_uniform,
            max_weight=self.cluster_sampling_max_weight,
        )

        if self.cluster_sampling_log_stats:
            print(
                "[cluster_sampling] "
                f"samples={int(stats['num_samples'])}, "
                f"missing={int(stats['num_missing_assignments'])} "
                f"({stats['missing_ratio']:.2%}), "
                f"clusters={int(stats['num_clusters_present'])}, "
                f"weight_range=[{stats['min_weight']:.4f}, {stats['max_weight']:.4f}], "
                f"mean={stats['mean_weight']:.4f}, std={stats['std_weight']:.4f}"
            )

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=self.cluster_sampling_replacement,
        )

    @staticmethod
    def split_fires(data_fold_id):
        """_summary_ Split the years into train/val/test set.

        Args:
            data_fold_id (_type_): _description_ Index of the respective split to choose, see method body for details.

        Returns:
            _type_: _description_
        """

        folds = [(2018, 2019, 2020, 2021),
                 (2018, 2019, 2021, 2020),
                 (2018, 2020, 2019, 2021),
                 (2018, 2020, 2021, 2019),
                 (2018, 2021, 2019, 2020),
                 (2018, 2021, 2020, 2019),
                 (2019, 2020, 2018, 2021),
                 (2019, 2020, 2021, 2018),
                 (2019, 2021, 2018, 2020),
                 (2019, 2021, 2020, 2018),
                 (2020, 2021, 2018, 2019),
                 (2020, 2021, 2019, 2018)]

        train_years = list(folds[data_fold_id][:2])
        val_years = list(folds[data_fold_id][2:3])
        test_years = list(folds[data_fold_id][3:4])

        print(
            f"Using the following dataset split:\nTrain years: {train_years}, Val years: {val_years}, Test years: {test_years}")

        return train_years, val_years, test_years
