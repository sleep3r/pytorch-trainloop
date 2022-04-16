import os
import numbers
from pathlib import Path

import cv2
import numpy as np
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED

imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
}

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imread(img_or_path, flag="color", channel_order="bgr"):
    """
    Reads an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is;
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
            Note that the `turbojpeg` backened does not support `unchanged`;
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, str):
        if not os.path.isfile(img_or_path):
            raise FileNotFoundError(f"img file does not exist: {img_or_path}")

        flag = imread_flags[flag] if isinstance(flag, str) else flag
        img = cv2.imread(img_or_path, flag)
        if flag == IMREAD_COLOR and channel_order == "rgb":
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img
    else:
        raise TypeError('"img" must be a numpy array or a str or a pathlib.Path object')


def imresize(img, size, return_scale=False, interpolation="bilinear", out=None):
    """
    Resizes image to a given size.

    Args:
        img (ndarray): The input image;
        size (tuple[int]): Target size (w, h);
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend;
        out (ndarray): The output destination.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or `resized_img`.
    """
    h, w = img.shape[:2]

    resized_img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode="constant"):
    """
    Pads the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded;
        shape (tuple[int]): Expected padding shape (h, w). Default: None;
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set;
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0;
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            - constant: pads with a constant value, this value is specified
                with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the
                last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result
                in [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with
                2 elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3].
    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(f"pad_val must be a int or a tuple. But received {type(pad_val)}")

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(f"Padding must be a int or a 2, or 4 element tuple. But received {padding}")

    # check padding mode
    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val,
    )
    return img
