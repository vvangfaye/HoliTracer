import cv2
import numpy as np
import random
import torch
from typing import Sequence


def normalize_function_mm(img):  # 更快
    # Mean values used to pre-training the pre-trained backbone models
    img = np.float32(img) if img.dtype != np.float32 else img.copy()

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    return torch.from_numpy(img.transpose((2, 0, 1))).float()


def normalize_function(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (np.ndarray): Image to be normalized, with shape (H, W, C).
        mean (ndarray or list): The mean for normalization.
        std (ndarray or list): The std for normalization.
        to_rgb (bool): Whether to convert to RGB format. Default is True.

    Returns:
        np.ndarray: The normalized image.
    """
    img = img / 255.0
    img = (img - mean) / std
    return img


def bgr2hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def photo_metric_distortion(
    img: np.ndarray,
    brightness_delta: int = 32,
    contrast_range: Sequence[float] = (0.5, 1.5),
    saturation_range: Sequence[float] = (0.5, 1.5),
    hue_delta: int = 18,
) -> np.ndarray:
    def convert(img: np.ndarray, alpha: int = 1, beta: int = 0) -> np.ndarray:
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            return convert(
                img, beta=random.uniform(-brightness_delta, brightness_delta)
            )
        return img

    def contrast(img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            return convert(
                img, alpha=random.uniform(contrast_range[0], contrast_range[1])
            )
        return img

    def saturation(img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            img_hsv = bgr2hsv(img)
            img_hsv[:, :, 1] = convert(
                img_hsv[:, :, 1],
                alpha=random.uniform(saturation_range[0], saturation_range[1]),
            )
            img = hsv2bgr(img_hsv)
        return img

    def hue(img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            img_hsv = bgr2hsv(img)
            img_hsv[:, :, 0] = (
                img_hsv[:, :, 0].astype(int) + random.randint(-hue_delta, hue_delta)
            ) % 180
            img = hsv2bgr(img_hsv)
        return img

    # random brightness
    img = brightness(img)

    # mode == 0 --> do random contrast first
    # mode == 1 --> do random contrast last
    mode = random.randint(0, 1)
    if mode == 1:
        img = contrast(img)

    # random saturation
    img = saturation(img)

    # random hue
    img = hue(img)

    # random contrast
    if mode == 0:
        img = contrast(img)

    return img
