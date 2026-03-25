from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import _T_co as T_co
import glob
import warnings
from .utils import get_means_stds_missing_values, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime


class FireSpreadDataset(Dataset):
    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                 crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                 stats_years: List[int], n_leading_observations_test_adjustment: Optional[int] = None,
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False, return_year: bool = False,
                 degree_encoding: str = "sin",
                 frontier_sampling_enabled: bool = False, frontier_band_width: int = 16,
                 frontier_boundary_ratio: float = 0.5, frontier_interior_ratio: float = 0.25,
                 frontier_hard_negative_ratio: float = 0.25, frontier_center_jitter: int = 8,
                 crop_search_trials: int = 10):
        """_summary_

        Args:
            data_dir (str): _description_ Root directory of the dataset, should contain several folders, each corresponding to a different fire.
            included_fire_years (List[int]): _description_ Years in dataset_root that should be used in this instance of the dataset.
            n_leading_observations (int): _description_ Number of days to use as input observation.
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF.
            is_train (bool): _description_ Whether this dataset is used for training or not. If True, apply geometric data augmentations. If False, only apply center crop to get the required dimensions.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            stats_years (List[int]): _description_ Which years to use for computing the mean and standard deviation of each feature. This is important for the test set, which should be standardized using the same statistics as the training set.
            n_leading_observations_test_adjustment (Optional[int], optional): _description_. Adjust the test set to look like it would with n_leading_observations set to this value.
        In practice, this means that if n_leading_observations is smaller than this value, some samples are skipped. Defaults to None. If None, nothing is skipped. This is especially used for the train and val set.
            features_to_keep (Optional[List[int]], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            degree_encoding (str, optional): Angular-feature encoding. "sin" preserves the current behavior,
              while "sincos" appends cosine channels for wind/aspect directions.
            frontier_sampling_enabled (bool, optional): Whether to bias training crops toward the current fire frontier.
            frontier_band_width (int, optional): Radius in pixels used to form the frontier band and near-front negative region.
            frontier_boundary_ratio (float, optional): Relative probability of sampling a boundary-centered crop when frontier sampling is enabled.
            frontier_interior_ratio (float, optional): Relative probability of sampling an interior-fire crop when frontier sampling is enabled.
            frontier_hard_negative_ratio (float, optional): Relative probability of sampling a near-front hard-negative crop when frontier sampling is enabled.
            frontier_center_jitter (int, optional): Random jitter applied around the sampled frontier point to preserve augmentation diversity.
            crop_search_trials (int, optional): Number of random crops to evaluate in the baseline fire-preferring crop selection.

        Raises:
            ValueError: _description_ Raised if input values are not in the expected ranges.
        """
        super().__init__()

        self.stats_years = stats_years
        self.return_doy = return_doy
        self.return_year = return_year
        self.degree_encoding = degree_encoding
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir
        self.frontier_sampling_enabled = frontier_sampling_enabled and is_train
        self.frontier_band_width = frontier_band_width
        self.frontier_boundary_ratio = frontier_boundary_ratio
        self.frontier_interior_ratio = frontier_interior_ratio
        self.frontier_hard_negative_ratio = frontier_hard_negative_ratio
        self.frontier_center_jitter = frontier_center_jitter
        self.crop_search_trials = crop_search_trials

        self.validate_inputs()

        # Compute how many samples to skip in the test set, to make it look like it would with n_leading_observations set to this value.
        if self.n_leading_observations_test_adjustment is None:
            self.skip_initial_samples = 0
        else:
            self.skip_initial_samples = self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.skip_initial_samples < 0:
                raise ValueError(f"n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations, but got {self.n_leading_observations_test_adjustment=} and {self.n_leading_observations=}")

        # Create an inventory of all images in the dataset, and how many data points each fire contains. Since we have multiple data points per fire,
        # we need to know how many data points each fire contains, to be able to map a dataset index to a specific fire.
        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum([sum(self.datapoints_per_fire[fire_year].values())
                          for fire_year in self.datapoints_per_fire])

        # Used in preprocessing and normalization. Better to define it once than build/call for every data point
        # The one-hot matrix is used for one-hot encoding of land cover classes
        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()

    def find_image_index_from_dataset_index(self, target_id) -> (int, str, int):
        """_summary_ Given the index of a data point in the dataset, find the corresponding fire that contains it,
        and its index within that fire.

        Args:
            target_id (_type_): _description_ Dataset index of the data point.

        Raises:
            RuntimeError: _description_ Raised if the dataset index is out of range.

        Returns:
            (int, str, int): _description_ Year, name of fire, index of data point within fire.
        """

        # Handle negative indexing, e.g. -1 should be the last item in the dataset
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(
                f"Tried to access item {target_id}, but maximum index is {self.length - 1}.")

        # The index is relative to the length of the full dataset. However, we need to make sure that we know which
        # specific fire the queried index belongs to. We know how many data points each fire contains from
        # self.datapoints_per_fire.
        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:
            if found_fire_year is None:
                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                    if target_id - first_id_in_current_fire < datapoints_in_fire:
                        found_fire_year = fire_year
                        found_fire_name = fire_name
                        break
                    else:
                        first_id_in_current_fire += datapoints_in_fire

        in_fire_index = target_id - first_id_in_current_fire

        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        """_summary_ Load the images corresponding to the specified data point from disk.

        Args:
            found_fire_year (_type_): _description_ Year of the fire that contains the data point.
            found_fire_name (_type_): _description_ Name of the fire that contains the data point.
            in_fire_index (_type_): _description_ Index of the data point within the fire.

        Returns:
            _type_: _description_ (x,y) or (x,y,doy) tuple, depending on whether return_doy is True or False.
            x is a tensor of shape (n_leading_observations, n_features, height, width), containing the input data.
            y is a tensor of shape (height, width) containing the binary next day's active fire mask.
            doy is a tensor of shape (n_leading_observations) containing the day of the year for each observation.
        """

        in_fire_index += self.skip_initial_samples
        end_index = (in_fire_index + self.n_leading_observations + 1)

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                if self.return_doy:
                    doys = f["data"].attrs["img_dates"][in_fire_index:(
                        end_index-1)]
                    doys = self.img_dates_to_doys(doys)
                    doys = torch.Tensor(doys)
            x, y = np.split(imgs, [-1], axis=0)
            # Last image's active fire mask is used as label, rest is input data
            y = y[0, -1, ...]
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    imgs.append(ds.read())
            x = np.stack(imgs[:-1], axis=0)
            y = imgs[-1][-1, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __getitem__(self, index):

        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(
            index)
        loaded_imgs = self.load_imgs(
            found_fire_year, found_fire_name, in_fire_index)

        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        x, y = self.preprocess_and_augment(x, y)

        # Remove duplicate static features, which can greatly reduce the number of features, since we use
        # one-hot encoded landcover types. The result would have different amounts of feature channels per
        # time step, therefore, we flatten the temporal dimension.
        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)

        # Discard features that we don't want to use
        elif self.features_to_keep is not None:
            if len(x.shape) != 4:
                raise NotImplementedError(f"Removing features is only implemented for 4D tensors, but got {x.shape=}.")
            x = x[:, self.features_to_keep, ...]

        # Build return tuple
        if self.return_year and self.return_doy:
            return x, y, doys, torch.tensor(found_fire_year, dtype=torch.long)
        elif self.return_year:
            return x, y, torch.tensor(found_fire_year, dtype=torch.long)
        elif self.return_doy:
            return x, y, doys
        else:
            return x, y

    def __len__(self):
        return self.length

    def get_sample_key(self, index: int) -> str:
        """
        Deterministic identifier for a dataset sample.

        Format:
            "{year}/{fire_name}/{effective_in_fire_index}"

        where effective_in_fire_index includes self.skip_initial_samples so keys
        match the actual time window used in load_imgs().
        """
        year, fire_name, in_fire_index = self.find_image_index_from_dataset_index(index)
        effective_index = in_fire_index + self.skip_initial_samples
        return f"{year}/{fire_name}/{effective_index}"

    def get_all_sample_keys(self) -> List[str]:
        """Return sample keys in dataset index order."""
        return [self.get_sample_key(i) for i in range(len(self))]

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError(
                "Returning day of year is only implemented for hdf5 files.")
        if self.n_leading_observations_test_adjustment is not None:
            if self.n_leading_observations_test_adjustment < self.n_leading_observations:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations.")
            if self.n_leading_observations_test_adjustment < 1:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to 1. Value 1 is used for having a single observation as input.")
        if self.degree_encoding not in {"sin", "sincos"}:
            raise ValueError("degree_encoding must be either 'sin' or 'sincos'.")
        if self.degree_encoding == "sincos" and self.remove_duplicate_features and self.n_leading_observations > 1:
            raise NotImplementedError(
                "degree_encoding='sincos' is not implemented together with remove_duplicate_features=True."
            )
        if self.frontier_sampling_enabled:
            total_ratio = (
                self.frontier_boundary_ratio
                + self.frontier_interior_ratio
                + self.frontier_hard_negative_ratio
            )
            if total_ratio <= 0:
                raise ValueError("Frontier sampling ratios must sum to a positive value.")
            if self.frontier_band_width < 1:
                raise ValueError("frontier_band_width must be at least 1.")
            if self.crop_search_trials < 1:
                raise ValueError("crop_search_trials must be at least 1.")

    def read_list_of_images(self):
        """_summary_ Create an inventory of all images in the dataset.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries.
            These dictionaries map names of fires that happened within the respective year to either
            a) the corresponding list of image files (in case hdf5 files are not used) or
            b) the individual hdf5 file for each fire.
        """
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()

                    imgs_per_fire[fire_year][fire_name] = fire_img_paths

                    if len(fire_img_paths) == 0:
                        warnings.warn(f"In dataset preparation: Fire {fire_year}: {fire_name} contains no images.",
                                      RuntimeWarning)
            else:
                fires_in_year = glob.glob(
                    f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        """_summary_ Compute how many data points each fire contains. This is important for mapping a dataset index to a specific fire.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries.
            The dictionaries map the fire name to the number of data points in that fire.
        """
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs) - self.skip_initial_samples
                else:
                    # Catch error case that there's no file
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"]) - self.skip_initial_samples
                # If we have two days of observations, and a lead of one day,
                # we can only predict the second day's fire mask, based on the first day's observation
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"In dataset preparation: Fire {fire_year}: {fire_name} does not contribute data points. It contains "
                        f"{len(fire_imgs)} images, which is too few for a lead of {self.n_leading_observations} observations.",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def standardize_features(self, x):
        """_summary_ Standardizes the input data, using the mean and standard deviation of each feature.
        Some features are excluded from this, which are the degree features (e.g. wind direction), and the land cover class.
        The binary active fire mask is also excluded, since it's added after standardization.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)

        Returns:
            _type_: _description_ Standardized input data, of shape (time_steps, features, height, width)
        """

        x = (x - self.means) / self.stds

        return x

    def img_dates_to_doys(self, img_dates):
        """_summary_ Convert a list of image dates to day of year values.

        Args:
            img_dates (_type_): _description_ List of strings, each string represents a date in the format YYYY-MM-DD.

        Returns:
            _type_: _description_ List of integers, each integer is the day of year corresponding to the respective date.
        """
        return [datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday for date in img_dates]

    def preprocess_and_augment(self, x, y):
        """_summary_ Preprocesses and augments the input data.
        This includes:
        1. Slight preprocessing of active fire features, if loading from TIF files.
        2. Geometric data augmentation.
        3. Applying sin to degree features, to ensure that the extreme degree values are close in feature space.
        4. Standardization of features.
        5. Addition of the binary active fire mask, as an addition to the fire mask that indicates the time of detection.
        6. One-hot encoding of land cover classes.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        x, y = self._prepare_loaded_sample(x, y)

        # Augmentation has to come before normalization, because we have to correct the angle features when we change
        # the orientation of the image.
        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        return self._finalize_preprocessed_sample(x, y)

    def preprocess_without_augmentation(self, x, y, fixed_crop_size: Optional[int] = None):
        """
        Deterministic preprocessing path for offline representation learning.

        If fixed_crop_size is provided, use a center crop of that size. Otherwise
        crop to the nearest multiple-of-32 shape like evaluation.
        """
        x, y = self._prepare_loaded_sample(x, y)

        if fixed_crop_size is not None:
            x, y = self.center_crop_fixed(x, y, fixed_crop_size)
        else:
            x, y = self.center_crop_x32(x, y)

        return self._finalize_preprocessed_sample(x, y)

    def _prepare_loaded_sample(self, x, y):
        x, y = torch.Tensor(x), torch.Tensor(y)

        # Preprocessing that has been done in HDF files already
        if not self.load_from_hdf5:

            # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
            # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
            # them with 0 instead.
            x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)

            # Turn active fire detection time from hhmm to hh.
            x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        y = (y > 0).long()
        return x, y

    def _finalize_preprocessed_sample(self, x, y):

        # Preserve the original angular values so we can optionally append cosine channels later.
        degree_values = x[:, self.indices_of_degree_features, ...].clone()

        # Some features take values in [0,360] degrees. By applying sin, we make sure that values near 0 and 360 are
        # close in feature space, since they are also close in reality.
        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(degree_values))

        # Compute binary mask of active fire pixels before normalization changes what 0 means.
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        x = self.standardize_features(x)

        # Adds the binary fire mask as an additional channel to the input data.
        x = torch.cat([x, binary_af_mask], axis=1)

        # Replace NaN values with 0, thereby essentially setting them to the mean of the respective feature.
        x = torch.nan_to_num(x, nan=0.0)

        # Create land cover class one-hot encoding, put it where the land cover integer was
        new_shape = (x.shape[0], x.shape[2], x.shape[3],
                     self.one_hot_matrix.shape[0])
        # -1 because land cover classes start at 1
        landcover_classes_flattened = x[:, 16, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.concatenate(
            [x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        if self.degree_encoding == "sincos":
            degree_cosines = torch.cos(torch.deg2rad(degree_values))
            degree_cosines = torch.nan_to_num(degree_cosines, nan=0.0)
            x = torch.cat([x, degree_cosines], dim=1)

        return x, y

    def _crop(self, x, y, top: int, left: int):
        x_crop = TF.crop(x, top, left, self.crop_side_length, self.crop_side_length)
        y_crop = TF.crop(y, top, left, self.crop_side_length, self.crop_side_length)
        return x_crop, y_crop

    def _max_crop_offsets(self, x: torch.Tensor):
        max_top = max(x.shape[-2] - self.crop_side_length, 0)
        max_left = max(x.shape[-1] - self.crop_side_length, 0)
        return max_top, max_left

    def _random_top_left(self, x: torch.Tensor):
        max_top, max_left = self._max_crop_offsets(x)
        top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
        left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        return top, left

    def _sample_fire_preferring_crop(self, x, y):
        best_n_fire_pixels = -1
        best_crop = (None, None)

        for _ in range(self.crop_search_trials):
            top, left = self._random_top_left(x)
            x_crop, y_crop = self._crop(x, y, top, left)

            # We really care about having fire pixels in the target. But if we don't find any there,
            # we care about fire pixels in the input, to learn to predict that no new observations will be made,
            # even though previous days had active fires.
            n_fire_pixels = x_crop[:, -1, ...].mean() + 1000 * y_crop.float().mean()
            if n_fire_pixels > best_n_fire_pixels:
                best_n_fire_pixels = n_fire_pixels
                best_crop = (x_crop, y_crop)

        return best_crop

    def _build_frontier_masks(self, x: torch.Tensor, y: torch.Tensor):
        prev_fire = (x[-1, -1, ...] > 0).float().unsqueeze(0).unsqueeze(0)
        if prev_fire.sum() == 0:
            return None

        dilated = F.max_pool2d(
            prev_fire,
            kernel_size=2 * self.frontier_band_width + 1,
            stride=1,
            padding=self.frontier_band_width,
        )
        eroded = 1.0 - F.max_pool2d(
            1.0 - prev_fire,
            kernel_size=2 * self.frontier_band_width + 1,
            stride=1,
            padding=self.frontier_band_width,
        )

        prev_fire_mask = prev_fire.squeeze(0).squeeze(0) > 0.5
        frontier_band = (dilated.squeeze(0).squeeze(0) - eroded.squeeze(0).squeeze(0)) > 0
        near_front = dilated.squeeze(0).squeeze(0) > 0
        y_mask = y > 0

        return {
            "boundary": frontier_band,
            "interior": prev_fire_mask,
            "hard_negative": near_front & (~prev_fire_mask) & (~y_mask),
        }

    def _sample_mask_center_crop(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        coords = torch.nonzero(mask, as_tuple=False)
        if len(coords) == 0:
            return None

        center_idx = np.random.randint(0, len(coords))
        center_y, center_x = coords[center_idx].tolist()
        if self.frontier_center_jitter > 0:
            center_y += np.random.randint(-self.frontier_center_jitter, self.frontier_center_jitter + 1)
            center_x += np.random.randint(-self.frontier_center_jitter, self.frontier_center_jitter + 1)

        max_top, max_left = self._max_crop_offsets(x)
        top = int(np.clip(center_y - self.crop_side_length // 2, 0, max_top))
        left = int(np.clip(center_x - self.crop_side_length // 2, 0, max_left))
        return self._crop(x, y, top, left)

    def _sample_frontier_crop(self, x: torch.Tensor, y: torch.Tensor):
        masks = self._build_frontier_masks(x, y)
        if masks is None:
            return None

        strategies = [
            ("boundary", self.frontier_boundary_ratio),
            ("interior", self.frontier_interior_ratio),
            ("hard_negative", self.frontier_hard_negative_ratio),
        ]

        # Prefer non-empty strategies but keep fallback order randomised.
        probs = np.array([weight for _, weight in strategies], dtype=np.float64)
        probs = probs / probs.sum()
        ordered_indices = list(np.random.choice(len(strategies), size=len(strategies), replace=False, p=probs))

        for idx in ordered_indices:
            strategy_name, _ = strategies[idx]
            crop = self._sample_mask_center_crop(x, y, masks[strategy_name])
            if crop is not None:
                return crop

        return None

    def augment(self, x, y):
        """_summary_ Applies geometric transformations:
          1. random square cropping, preferring images with a) fire pixels in the output and b) (with much less weight) fire pixels in the input
          2. rotate by multiples of 90°
          3. flip horizontally and vertically
        Adjustment of angles is done as in https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/image_utils.py

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        # Need square crop to prevent rotation from creating/destroying data at the borders, due to uneven side lengths.
        if self.frontier_sampling_enabled:
            crop = self._sample_frontier_crop(x, y)
            if crop is None:
                crop = self._sample_fire_preferring_crop(x, y)
            x, y = crop
        else:
            x, y = self._sample_fire_preferring_crop(x, y)

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = 360 - \
                x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)

            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (x[:, self.indices_of_degree_features,
                                                          ...] - 90 * rotate) % 360

        return x, y

    def center_crop_x32(self, x, y):
        """_summary_ Crops the center of the image to side lengths that are a multiple of 32,
        which the ResNet U-net architecture requires. Only used for computing the test performance.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        T, C, H, W = x.shape
        H_new = H//32 * 32
        W_new = W//32 * 32

        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        return x, y

    def center_crop_fixed(self, x, y, side_length: int):
        """
        Center crop to a fixed square size for deterministic offline feature extraction.
        """
        x = TF.center_crop(x, (side_length, side_length))
        y = TF.center_crop(y, (side_length, side_length))
        return x, y

    def flatten_and_remove_duplicate_features_(self, x):
        """_summary_ For a simple U-Net, static and forecast features can be removed everywhere but in the last time step
        to reduce the number of features. Since that would result in different numbers of channels for different
        time steps, we flatten the temporal dimension.
        Also discards features that we don't want to use.

        Args:
            x (_type_): _description_ Input tensor data of shape (n_leading_observations, n_features, height, width)

        Returns:
            _type_: _description_
        """
        static_feature_ids, dynamic_feature_ids = self.get_static_and_dynamic_features_to_keep(self.features_to_keep)
        dynamic_feature_ids = torch.tensor(dynamic_feature_ids).int()

        x_dynamic_only = x[:-1, dynamic_feature_ids, :, :].flatten(start_dim=0, end_dim=1)
        x_last_day = x[-1, self.features_to_keep, ...].squeeze(0)

        return torch.cat([x_dynamic_only, x_last_day], axis=0)

    @staticmethod
    def get_static_and_dynamic_feature_ids():
        """_summary_ Returns the indices of static and dynamic features.
        Static features include topographical features and one-hot encoded land cover classes.

        Returns:
            _type_: _description_ Tuple of lists of integers, first list contains static feature indices, second list contains dynamic feature indices.
        """
        static_feature_ids = [12,13,14] + list(range(16,33))
        dynamic_feature_ids = list(range(12)) + [15] + list(range(33,40))
        return static_feature_ids, dynamic_feature_ids

    @staticmethod
    def get_static_and_dynamic_features_to_keep(features_to_keep:Optional[List[int]]):
        """_summary_ Returns the indices of static and dynamic features that should be kept, based on the input list of feature indices to keep.

        Args:
            features_to_keep (Optional[List[int]]): _description_

        Returns:
            _type_: _description_
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_feature_ids()

        if type(features_to_keep) == list:
            dynamic_features_to_keep = list(set(dynamic_features_to_keep) & set(features_to_keep))
            dynamic_features_to_keep.sort()

        if type(features_to_keep) == list:
            static_features_to_keep = list(set(static_features_to_keep) & set(features_to_keep))
            static_features_to_keep.sort()

        return static_features_to_keep, dynamic_features_to_keep

    @staticmethod
    def get_n_features(n_leading_observations: int, features_to_keep: Optional[List[int]], remove_duplicate_features: bool,
                       degree_encoding: str = "sin"):
        """_summary_ Static method to compute how many feature channels a sample will have in a dataset with the given parameters.

        Args:
            n_leading_observations (int): _description_
            features_to_keep (Optional[List[int]]): _description_
            remove_duplicate_features (bool): _description_

        Returns:
            _type_: _description_
        """

        # Full dataset has 40 features.
        # Features that we are not able to use due to lack of ground truth data from data leakage are taken out already.
        # Static features are one-hot encoded land cover types, topography and fuel amount
        n_static = 20
        n_dynamic = 20

        if features_to_keep is not None:
            static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_features_to_keep(features_to_keep)
            n_static = len(static_features_to_keep)
            n_dynamic = len(dynamic_features_to_keep)
        elif degree_encoding == "sincos":
            # Appending cosine channels adds one static direction feature (aspect)
            # and two dynamic direction features (observed + forecast wind direction).
            n_static += 1
            n_dynamic += 2

        if n_leading_observations < 1:
            raise ValueError(
                f"Need at least one day of observations, but got {n_leading_observations=}.")

        if remove_duplicate_features and n_leading_observations > 1:
            return n_dynamic * (n_leading_observations - 1) + n_dynamic + n_static

        return (n_dynamic + n_static) * n_leading_observations
