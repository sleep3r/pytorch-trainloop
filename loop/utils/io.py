import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def aggregate_masks(masks: list) -> np.ndarray:
    mask_mean = masks[0] & masks[1] & masks[2]
    return mask_mean


def generate_uncertainty_mask(masks: list) -> np.ndarray:
    mask_uncertainty = (masks[0] ^ masks[1]) | (masks[1] ^ masks[2])
    return mask_uncertainty


def read_dataset(
        images_kind: str,
        images_path: Path,
        masks_path: Path,
        subsample: np.ndarray,
        projections: List[str],
        aggregate: bool = True
) -> (List[str], List[Path], List[Optional[Path]]):
    """
    Reads the dataset and returns the image names and the corresponding masks.

    Args:
        images_kind: kind of images to read;
        images_path: path to the images;
        masks_path: path to the masks;
        subsample: array of indexes to subsample for cross-validation;
        projections: list of projection types to filter by;
        aggregate: whether to aggregate the masks or not.
    Returns:
        img_names: list of image names;
        images_fps: list of image file paths;
        masks_fps: list of mask file paths.
    """
    img_names = [img.stem for img in images_path.iterdir()]

    # filtering by projection type
    img_names = [*filter(lambda x: x.split("_")[1] in projections, img_names)]

    # filtering for cross-validation
    # here int(x.split("_")[0]) is the index of the data image
    img_names = [*filter(lambda x: int(x.split("_")[0]) in subsample, img_names)]

    # image paths
    images: List[Path] = [next(images_path.glob(f"{name}.png")) for name in img_names]

    if images_kind == "pneumonia":
        # we assume three or less (some expert found no pathology) masks for the image here
        masks: List[List[Optional[Path]]] = [sorted(masks_path.glob(f"./expert*/{name}.*")) for name in img_names]
    elif images_kind == "norma":
        # reading norma, so no masks here, we will generate them later in __getitem__
        masks = [[None] for _ in range(len(img_names))]
    else:
        raise KeyError(f"WTF kind?! {images_kind}")

    assert len(masks) == len(images) == len(img_names)

    if aggregate and images_kind == "pneumonia":
        # should filter images with an incomplete set of masks (< 3)
        img_names, images, masks = filter_dataset(img_names, images, masks)
    elif not aggregate and images_kind == "pneumonia":
        # adding nones for the images with an incomplete set of masks (< 3)
        for i, img_masks in enumerate(masks):
            if len(img_masks) != 3:
                for _ in range(3 - len(img_masks)):
                    masks[i].append(None)

        # flattening results, cause no need of aggregation
        images = [images[i] for i in range(len(img_names)) for _ in range(3)]
        masks = [[masks[i][j]] for i in range(len(img_names)) for j in range(3)]
    return img_names, images, masks


def filter_dataset(img_names: list, images_fps: list, masks_fps: list) -> (list, list):
    """Filters data with missing expert markup, that's important for aggregation"""
    bad_images = []
    for i, masks in enumerate(masks_fps):
        if len(masks) != 3:
            bad_images.append(i)

    img_names = [img_names[i] for i in range(len(img_names)) if i not in bad_images]
    images_fps = [images_fps[i] for i in range(len(images_fps)) if i not in bad_images]
    masks_fps = [masks_fps[i] for i in range(len(masks_fps)) if i not in bad_images]
    return img_names, images_fps, masks_fps


def read_duplicates(duplicates_path: str) -> dict:
    """
    Reads duplicates info for cross-validation.

    Args:
        duplicates_path (str): path to the duplicates json file.
    Returns:
        dict: duplicates, where keys are image indexes and values groups indexes.
    """
    with open(duplicates_path, "r") as f:
        groups = json.load(f)

    data_indexes = [*map(int, groups.keys())]

    assert data_indexes[0] == 1, "duplicates data indexes should start from 1"
    assert data_indexes == [*range(1, max(data_indexes) + 1)], "duplicates data indexes should be full and consecutive"
    return groups


def read_norma_ids(images_path: str) -> List[int]:
    """Reads norma images ids for further use in cv splitting."""
    images_path = Path(images_path)
    norma_images_path = images_path / "norma"

    norma_idx = [int(img.stem.split("_")[0]) for img in norma_images_path.iterdir()]
    return norma_idx
