import albumentations as albu
import torchvision.transforms.functional as TF
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

from config import CfgDict


class ToTensor(DualTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        im_tensor = TF.to_tensor(img)
        return im_tensor

    def apply_to_mask(self, mask, **params):
        mask_tensor = TF.to_tensor(mask)
        return mask_tensor


class Normalize(ImageOnlyTransform):
    def __init__(self, image_params: CfgDict, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.img_norm_cfg = image_params.img_norm_cfg

    def apply(self, img, **params):
        img_norm = TF.normalize(img, mean=self.img_norm_cfg.mean, std=self.img_norm_cfg.std)
        return img_norm


class UnNormalize(object):
    """Useful class for further use in visualization needs"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def post_transform(image_params: CfgDict):
    return [
        albu.Resize(height=image_params.height, width=image_params.width, always_apply=True),
        ToTensor(),
        Normalize(image_params=image_params)
    ]
