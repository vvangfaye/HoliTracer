import os
import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .utils import normalize_function, normalize_function_mm, photo_metric_distortion


class MVTrainDataset(Dataset):
    """Multiview Dataset for loading images and labels from an HDF5 file."""

    def __init__(self, hdf5_path, training=False):
        """Initializes the dataset with images and labels from the HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file containing the dataset.
            training (bool): If True, data augmentation is applied.
        """
        super(MVTrainDataset, self).__init__()
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        if not h5py.is_hdf5(hdf5_path):
            raise ValueError(f"Invalid HDF5 file: {hdf5_path}")

        hf = h5py.File(hdf5_path, "r")
        self.image1 = hf["image1"]
        self.image2 = hf["image2"]
        self.image3 = hf["image3"]
        self.mask = hf["label"]
        self.training = training
        
        self.std = [0.229, 0.224, 0.225]
        # self.std = [i * 255 for i in self.std]
        self.mean = [0.485, 0.456, 0.406]
        # self.mean = [i * 255 for i in self.mean]

    def __len__(self):
        """Returns the total number of samples."""
        return self.mask.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and applies transformations if training.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the images and the mask tensor.
        """
        img1 = self.image1[idx]
        img2 = self.image2[idx]
        img3 = self.image3[idx]
        ann = self.mask[idx]

        if self.training:
            img1, img2, img3, ann = self.augment(img1, img2, img3, ann)

        img1 = normalize_function_mm(img1)
        img2 = normalize_function_mm(img2)
        img3 = normalize_function_mm(img3)
        
        out = {
            "img1": img1,
            "img2": img2,
            "img3": img3,
            "mask": torch.from_numpy(ann).long(),
        }

        return out

    @staticmethod
    def augment(img1, img2, img3, ann):
        """Applies data augmentation techniques to the images and annotation.

        Args:
            img1 (np.ndarray): First image.
            img2 (np.ndarray): Second image.
            img3 (np.ndarray): Third image.
            ann (np.ndarray): Annotation mask.

        Returns:
            tuple: Augmented images and annotation mask.
        """
        # def color_distortion(image):
        #    """Applies random brightness, contrast, and saturation adjustments."""
        #    alpha = 1.0 + np.random.uniform(-0.5, 0.5)  # Brightness
        #    beta = 0.5 + np.random.uniform(-0.5, 0.5)   # Contrast
        #    saturation_scale = 0.5 + np.random.uniform(0, 1)  # Saturation

        #    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        #    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        #    distorted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #    return distorted_image

        # def gaussian_noise(image):
        #    """Adds Gaussian noise to the image."""
        #    row, col, ch = image.shape
        #    mean = 0
        #    var = 0.1
        #    sigma = var ** 0.5
        #    gauss = np.random.normal(mean, sigma, (row, col, ch))
        #    noisy = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)
        #    return noisy

        # def random_cover(image):
        #    """Covers a random region of the image with zeros."""
        #    row, col, ch = image.shape
        #    r = random.randint(8, row // 8)
        #    c = random.randint(8, col // 8)
        #    m = np.zeros((r, c, ch), dtype=image.dtype)
        #    sr = random.randint(0, row - r - 1)
        #    sc = random.randint(0, col - c - 1)
        #    image[sr:sr + r, sc:sc + c, :] = m
        #    return image

        if random.random() < 0.5:
            flip_type = random.choice([0, 1])
            img1 = cv2.flip(img1, flip_type)
            img2 = cv2.flip(img2, flip_type)
            img3 = cv2.flip(img3, flip_type)
            ann = cv2.flip(ann, flip_type)

        if random.random() < 0.5:
            img1 = photo_metric_distortion(img1)
            img2 = photo_metric_distortion(img2)
            img3 = photo_metric_distortion(img3)

        # if random.random() < 0.5:
        #    img1 = photo_metric_distortion(img1)
        #    img2 = photo_metric_distortion(img2)
        #    img3 = photo_metric_distortion(img3)

        # Uncomment below to enable additional augmentations
        # if random.random() < 0.4:
        #     img1 = color_distortion(img1)
        #     img2 = color_distortion(img2)
        #     img3 = color_distortion(img3)
        # if random.random() < 0.3:
        #     img1 = gaussian_noise(img1)
        #     img2 = gaussian_noise(img2)
        #     img3 = gaussian_noise(img3)

        # if random.random() < 0.5:
        #    img1 = random_cover(img1)
        # if random.random() < 0.2:
        #     img2 = random_cover(img2)
        # if random.random() < 0.2:
        #     img3 = random_cover(img3)

        return img1, img2, img3, ann

class MVInferDataset(Dataset):
    def __init__(self, hdf5_path, args):
        """
        Initialize the dataset.

        Parameters:
            hdf5_path (str): Path to the HDF5 file containing the dataset.
            args (Namespace or dict): Arguments containing configuration parameters.
                Expected keys:
                    - nclass (int): Number of segmentation classes.
        """
        super(MVInferDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.args = args if isinstance(args, dict) else vars(args)
        self.nclass = self.args.get('nclass', 2)

        # Initialize normalization (same as used during training)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Placeholders for HDF5 file and metadata
        self.h5_file = None
        self.length = None
        self.image_names = None
        self.image_sizes = None

    def __len__(self):
        """
        Return the total number of patches in the dataset.
        """
        if self.length is None:
            # Open HDF5 file in read mode to get the length
            with h5py.File(self.hdf5_path, 'r') as h5_file:
                self.length = h5_file['image1'].shape[0]
        return self.length

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'img1': Tensor of shape (3, H, W) for the first scale.
                - 'img2': Tensor of shape (3, H, W) for the second scale.
                - 'img3': Tensor of shape (3, H, W) for the third scale.
                - 'image_index': Integer index of the original image.
                - 'i': Integer y-coordinate of the patch's top-left corner.
                - 'j': Integer x-coordinate of the patch's top-left corner.
        """
        # Open the HDF5 file if not already open
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

            # Load image names and sizes once
            self.image_names = [name.decode('utf-8') for name in self.h5_file['image_names'][:]]
            self.image_sizes = self.h5_file['image_sizes'][:]

        # Retrieve multi-scale images and metadata
        img1 = self.h5_file['image1'][idx]  # Shape: (3, H, W), uint8
        img2 = self.h5_file['image2'][idx]  # Shape: (3, H, W), uint8
        img3 = self.h5_file['image3'][idx]  # Shape: (3, H, W), uint8
        image_index = self.h5_file['image_index'][idx]  # Integer
        i = self.h5_file['i'][idx]  # Integer
        j = self.h5_file['j'][idx]  # Integer

        # Convert images to float tensors and normalize
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img3 = self.to_tensor(img3)

        sample = {
            'img1': img1,  # Tensor: (3, H, W)
            'img2': img2,  # Tensor: (3, H, W)
            'img3': img3,  # Tensor: (3, H, W)
            'image_index': torch.tensor(image_index, dtype=torch.long),
            'i': torch.tensor(i, dtype=torch.long),
            'j': torch.tensor(j, dtype=torch.long),
        }

        return sample

    def to_tensor(self, img):
        """
        Converts a NumPy image array to a normalized PyTorch tensor.

        Parameters:
            img (numpy.ndarray): Image array of shape (3, H, W), dtype=uint8.

        Returns:
            torch.Tensor: Normalized tensor of shape (3, H, W), dtype=torch.float32.
        """
        # Convert from uint8 to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to PyTorch tensor
        img = torch.from_numpy(img)  # Shape: (3, H, W)

        # Normalize
        img = self.normalize(img)

        return img

    def get_image_info(self):
        """
        Retrieves image names and sizes.

        Returns:
            tuple: (image_names, image_sizes)
        """
        if self.h5_file is None:
            with h5py.File(self.hdf5_path, 'r') as h5_file:
                image_names = [name.decode('utf-8') for name in h5_file['image_names'][:]]
                image_sizes = h5_file['image_sizes'][:]
        else:
            image_names = self.image_names
            image_sizes = self.image_sizes
        return image_names, image_sizes

    def close(self):
        """
        Closes the HDF5 file if it's open.
        """
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """
        Destructor to ensure the HDF5 file is closed.
        """
        self.close()

class SVTrainDataset(Dataset):
    """Single View Dataset for loading images and labels from an HDF5 file."""

    def __init__(self, hdf5_path, training=False):
        """Initializes the dataset with images and labels from the HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file containing the dataset.
            training (bool): If True, data augmentation is applied.
        """
        super(SVTrainDataset, self).__init__()
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        if not h5py.is_hdf5(hdf5_path):
            raise ValueError(f"Invalid HDF5 file: {hdf5_path}")

        hf = h5py.File(hdf5_path, "r")
        self.image = hf["image"]  # Changed to load single image dataset
        self.mask = hf["label"]
        self.training = training

        self.std = [0.229, 0.224, 0.225]
        # self.std = [i * 255 for i in self.std]
        self.mean = [0.485, 0.456, 0.406]
        # self.mean = [i * 255 for i in self.mean]

    def __len__(self):
        """Returns the total number of samples."""
        return self.mask.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and applies transformations if training.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image and the mask tensor.
        """
        img = self.image[idx]
        ann = self.mask[idx]

        if self.training:
            img, ann = self.augment(img, ann)  # Modified augment call

        img = normalize_function_mm(img) # Normalize single image

        out = {
            "img": img,  # Changed key to "img" for single view
            "mask": torch.from_numpy(ann).long(),
        }

        return out

    @staticmethod
    def augment(img, ann):
        """Applies data augmentation techniques to the image and annotation.

        Args:
            img (np.ndarray): Single image.
            ann (np.ndarray): Annotation mask.

        Returns:
            tuple: Augmented image and annotation mask.
        """

        if random.random() < 0.5:
            flip_type = random.choice([0, 1])
            img = cv2.flip(img, flip_type)
            ann = cv2.flip(ann, flip_type)

        if random.random() < 0.5:
            img = photo_metric_distortion(img)

        return img, ann # Return single image and annotation