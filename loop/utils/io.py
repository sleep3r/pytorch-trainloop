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
        per_folder_ratio: float,
        aggregate: bool = True
) -> (list, list):
    """
    Reads the dataset and returns the image names and the corresponding masks.

    Args:
        images_kind: kind of images to read;
        images_path: path to the images;
        masks_path: path to the masks;
        per_folder_ratio: ratio of images to read per folder;
        aggregate: whether to aggregate the masks or not.
    Returns:
        img_names: list of image names;
        images_fps: list of image file paths;
        masks_fps: list of mask file paths.
    """
    img_names = [img.stem for img in images_path.iterdir()]
    img_names = img_names[: int(len(img_names) * per_folder_ratio)]

    images: List[Path] = [next(images_path.glob(f"{name}.png")) for name in img_names]

    if images_kind == "pneumonia":
        masks: List[List[Optional[Path]]] = [sorted(masks_path.glob(f"./expert*/{name}.*")) for name in img_names]
    else:
        # we will generate masks later in __getitem__
        masks = [[None] for _ in range(len(img_names))]

    assert len(masks) == len(images) == len(img_names)

    if aggregate and images_kind == "pneumonia":
        img_names, images, masks = filter_dataset(img_names, images, masks)
    elif not aggregate and images_kind == "pneumonia":
        # adding missing masks with no expert markup
        for i, img_masks in enumerate(masks):
            if len(img_masks) != 3:
                for _ in range(3 - len(img_masks)):
                    masks[i].append(None)

        images = [images[i] for i in range(len(img_names)) for _ in range(3)]
        masks = [[masks[i][j]] for i in range(len(img_names)) for j in range(3)]
    return img_names, images, masks


def filter_dataset(img_names: list, images_fps: list, masks_fps: list) -> (list, list):
    # filter masks with no expert markup, that's important for aggregation
    bad_images = []
    for i, masks in enumerate(masks_fps):
        if len(masks) != 3:
            bad_images.append(i)

    img_names = [img_names[i] for i in range(len(img_names)) if i not in bad_images]
    images_fps = [images_fps[i] for i in range(len(images_fps)) if i not in bad_images]
    masks_fps = [masks_fps[i] for i in range(len(masks_fps)) if i not in bad_images]
    return img_names, images_fps, masks_fps
