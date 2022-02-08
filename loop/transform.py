import albumentations as albu

from config import CfgDict
from utils.aug import post_transform

__all__ = ["ChestTransform"]


def augmentations():
    channel_augs = [
        # albu.HueSaturationValue(p=0.5),
        # albu.ChannelShuffle(p=0.5),
    ]
    result = [
        # albu.OneOf(channel_augs, p=0.25),
        # albu.OneOf([
        #     albu.CLAHE(clip_limit=2, p=0.25),
        #     albu.Sharpen(p=0.25),
        #     albu.Emboss(p=0.25),
        # ], p=0.25),
        # albu.RandomBrightnessContrast(
        #     brightness_limit=0.4,
        #     contrast_limit=0.5,
        #     p=0.25
        # ),
        # albu.RandomGamma(gamma_limit=(85, 115), p=0.25),
        # albu.ImageCompression(quality_lower=60, quality_upper=99, p=0.25),
    ]
    return result


def augmentations_light():
    result = [
        albu.OneOf([
            albu.GaussNoise(),
        ], p=0.2),
        albu.OneOf([
            albu.Sharpen(),
            albu.Emboss(),
        ], p=0.2),
        albu.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.5,
            p=0.25
        ),
        albu.ImageCompression(quality_lower=80, quality_upper=99, p=0.25),
    ]
    return result


class ChestTransform:
    def __init__(
            self,
            image_params: CfgDict,
            kind: str = "train",
            augs_lvl: str = "light",
    ):
        self.image_params = image_params
        self.kind = kind
        self.augs_lvl = augs_lvl

    def get_transforms(self):
        if self.kind == "train":
            transforms = train_transform(
                image_params=self.image_params,
                augs_lvl=self.augs_lvl,
            )
        elif self.kind == "val":
            transforms = valid_transform(self.image_params)
        else:
            return infer_transform(self.image_params)
        return transforms


def train_transform(
        image_params: CfgDict,
        augs_lvl: str = "light",
):
    if augs_lvl == "hard":
        transforms = augmentations()
    elif augs_lvl == "light":
        transforms = augmentations_light()
    else:
        raise ValueError("Incorrect `augs_lvl`")

    result = albu.Compose(
        transforms=[
            *transforms,
            *post_transform(image_params)
        ]
    )
    return result


def valid_transform(image_params: CfgDict):
    return albu.Compose(
        transforms=[
            *post_transform(image_params)
        ]
    )


def infer_transform(image_params: CfgDict):
    return albu.Compose(
        transforms=[
            *post_transform(image_params)
        ]
    )
