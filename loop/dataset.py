import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data import Dataset

from config import CfgDict, object_from_dict
from utils.distributed import get_dist_info
from utils.image import imread
from utils.io import read_dataset, aggregate_masks, generate_uncertainty_mask
from utils.logging import status, print_log

__all__ = ["ChestDataset", "Collect", "OrderedDistributedSampler"]


class ChestDataset(Dataset):
    """
    Base dataset class.

    Args:
        image_params: image parameters;
        images_path: path to images;
        masks_path: path to masks;
        kind: dataset kind: train or val;
        subsamples: pneumonia and norma ids to load;
    """
    CLASSES = ["background", "pneumonia", "uncertainty"]

    def __init__(
            self,
            image_params: CfgDict,
            images_path: str,
            masks_path: str,
            kind: str,
            projections: List[str],
            subsamples: Tuple[np.ndarray, np.ndarray],
            classes: List[str],
            batch_size: int = None,
            steps_per_epoch: int = None,
            aggregate_masks: bool = True,
            read_norma: bool = False,
            transform: CfgDict = None
    ):
        self.kind = kind
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.image_params = image_params
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.subsamples = subsamples
        self._classes = classes
        self.projections = projections
        self.aggregate_masks = aggregate_masks
        self.read_norma = read_norma
        self.transform = transform

        self.local_rank, self.world_size = get_dist_info()

        self._init_dataset()

    def __len__(self):
        if self.kind == "train" and self.steps_per_epoch is not None:
            return self.steps_per_epoch * self.batch_size * self.world_size
        return len(self.dataset)

    @property
    def dataset(self):
        return self._dataset

    def _init_dataset(self):
        with status(f"[bold green] ({self.kind}) Dataset init\n"):
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in self._classes]
            self.augmentation = object_from_dict(self.transform).get_transforms()  # noqa

            self.unique_names, self.images_fps, self.masks_fps = self._read(
                images_kind="pneumonia",
                subsample=self.subsamples[0]
            )

            len_norma = 0
            if self.read_norma:
                self.unique_names_norma, self.images_fps_norma, self.masks_fps_norma = self._read(
                    images_kind="norma",
                    subsample=self.subsamples[1]
                )
                len_norma = len(self.unique_names_norma)

                self.unique_names += self.unique_names_norma
                self.images_fps += self.images_fps_norma
                self.masks_fps += self.masks_fps_norma

            self._dataset = [*zip(self.images_fps, self.masks_fps)]

        assert len(self._dataset) != 0, f"({self.kind}) Dataset is empty!"
        print_log(f"({self.kind}) Final len of dataset: {len(self._dataset)} ({len_norma} norma)\n", logger="train")

    def _read(self, images_kind: str, subsample: np.ndarray) -> (list, list, list):
        images_path = self.images_path / images_kind

        unique_names, images_fps, masks_fps = read_dataset(
            images_kind,
            images_path,
            self.masks_path,
            subsample,
            self.projections,
            aggregate=self.aggregate_masks,
        )
        return unique_names, images_fps, masks_fps

    def __getitem__(self, i):
        image_path = str(self._dataset[i][0])
        image: np.ndarray = imread(image_path)

        # 3 mask_fps if doing aggregation, 1 mask_fps otherwise
        # None instead of mask_fps in case of norma img
        markup_paths: List[Optional[Path]] = self._dataset[i][1]

        # aggregation from 3 masks
        if self.aggregate_masks and any(markup_paths):
            masks = [np.load(str(mask_path))["mask"] for mask_path in markup_paths]
            mask = aggregate_masks(masks)
            uncertainty_mask = generate_uncertainty_mask(masks)
            label = 1

        # no aggregation, reading just one mask
        elif not self.aggregate_masks and any(markup_paths):
            mask = np.load(str(markup_paths[0]))["mask"]
            uncertainty_mask = None
            label = 1

        # norma blank mask generation
        elif not any(markup_paths):
            mask = np.zeros(image.shape[:2])
            uncertainty_mask = None
            label = 0

        else:
            raise RuntimeError(f"WTF? Masks generation failed!")

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        result = self.augmentation(image=image, mask=mask)

        if uncertainty_mask is not None:
            unc_result = self.augmentation(image=image, mask=uncertainty_mask)
            uncertainty_mask = unc_result["mask"]

        # int label for further optional classification
        result["label"] = label

        result["path"] = image_path
        result["uncertainty_mask"] = uncertainty_mask
        return result


class Collect:
    def __init__(self, meta_keys: List[str]):
        """
        Args:
            meta_keys: list of keys to collect from meta.
        """
        self.meta_keys = meta_keys

    def __call__(self, batch):
        result = dict()

        result["image"] = torch.stack([item["image"] for item in batch])
        result["mask"] = torch.stack([item["mask"] for item in batch])
        result["label"] = torch.FloatTensor([item["label"] for item in batch])

        result["image_meta"] = [{key: item["image_meta"][key] for key in self.meta_keys} for item in batch]
        return result


class OrderedDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    (!) Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        super().__init__(data_source=dataset)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples
