import torch
import cv2
import numpy as np

def normalize_function_mm(img, device):
    # Mean values used to pre-training the pre-trained backbone model
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
    tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
    
    return tensor.unsqueeze(0).to(device)


def normalize_function(img, device):
    # Mean values used to pre-training the pre-trained backbone models
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    results = mmcv.imnormalize(img, mean, std, to_rgb=True)
    tensor = torch.from_numpy(results.transpose((2, 0, 1))).float()
    return tensor.unsqueeze(0).to(device)
