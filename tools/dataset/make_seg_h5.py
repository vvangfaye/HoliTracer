import os
import numpy as np
from PIL import Image
import cv2
import math
import h5py
import argparse
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def crop_patch(image, label, start_coord=(0, 0), size=512):
    """
    Crops a patch from the given image and label starting at specified coordinates.

    Parameters:
        image (ndarray): The input image.
        label (ndarray): The corresponding label.
        start_coord (tuple): The starting (x, y) coordinates for cropping.
        size (int): The size of the patch to crop.

    Returns:
        tuple: The cropped image patch and label patch.
    """
    x, y = start_coord
    return image[x : x + size, y : y + size, :], label[x : x + size, y : y + size]


def create_hdf5(image_path, label_path, output_hdf5, downsample_factors, size=512):
    """
    Creates an HDF5 file containing multi-scale image patches and corresponding labels.

    Parameters:
        image_path (str): The directory path containing images.
        label_path (str): The directory path containing labels.
        output_hdf5 (str): The path to the output HDF5 file.
        downsample_factors (tuple): A tuple of three downsample factors.
        size (int): The size of the patches to crop.

    Returns:
        None
    """
    print(f"==============[{downsample_factors}]==============")
    assert (
        len(downsample_factors) == 3
    ), "Downsample factors must be a tuple of length 3."
    d1, d2, d3 = downsample_factors
    print(f"Image Path: {image_path}")
    print(f"Label Path: {label_path}")
    print(f"Output HDF5 File: {output_hdf5}")
    if not os.path.exists(os.path.dirname(output_hdf5)):
        os.makedirs(os.path.dirname(output_hdf5))

    image_names = os.listdir(image_path)
    image_names = [
        name
        for name in image_names
        if name.endswith(".jpg") and not name.startswith(".")
    ]
    num_images = len(image_names)
    first_image_name = image_names[0]
    image_sample = cv2.imread(os.path.join(image_path, first_image_name))
    height, width, _ = image_sample.shape

    s1 = d1 * size
    s2 = d2 * size
    s3 = d3 * size
    pad_size = (s3 - s1) // 2
    delta23 = (s3 - s2) // 2

    num_patches_per_image = math.ceil(height / s1) * math.ceil(width / s1)
    total_patches = num_patches_per_image * num_images

    with h5py.File(output_hdf5, "w") as hf:
        image1_dataset = hf.create_dataset(
            "image1", shape=(total_patches, size, size, 3), dtype="uint8"
        )
        image2_dataset = hf.create_dataset(
            "image2", shape=(total_patches, size, size, 3), dtype="uint8"
        )
        image3_dataset = hf.create_dataset(
            "image3", shape=(total_patches, size, size, 3), dtype="uint8"
        )
        label_dataset = hf.create_dataset(
            "label", shape=(total_patches, size, size), dtype="uint8"
        )
        idx = 0
        for image_name in tqdm(image_names):
            name, _ = os.path.splitext(image_name)
            image0 = cv2.imread(os.path.join(image_path, name + ".jpg"))
            label0 = cv2.imread(
                os.path.join(label_path, name + ".png"), cv2.IMREAD_GRAYSCALE
            ).astype(np.uint8)
            image_padded = np.pad(
                image0,
                pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                mode="constant",
            )
            # If the maximum label value is greater than 200, set all positive values to 1
            if np.max(label0) > 200:
                label0[label0 > 0] = 1

            for i in range(0, height, s1):
                i = height - s1 if i + s1 > height else i
                for j in range(0, width, s1):
                    j = width - s1 if j + s1 > width else j
                    img, ann = crop_patch(
                        image=image0, label=label0, start_coord=(i, j), size=size
                    )
                    image1_dataset[idx] = img
                    image2_dataset[idx] = cv2.resize(
                        image_padded[
                            i + delta23 : i + delta23 + s2,
                            j + delta23 : j + delta23 + s2,
                            :,
                        ],
                        (size, size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    image3_dataset[idx] = cv2.resize(
                        image_padded[i : i + s3, j : j + s3, :],
                        (size, size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    label_dataset[idx] = ann
                    idx += 1


def parse_args():

    parser = argparse.ArgumentParser(
        description="Create an multi-view HDF5 dataset from images and labels."
    )
    parser.add_argument(
        "--view_size", type=int, default=512, help="The size of the patches to crop."
    )
    parser.add_argument(
        "--downsample_factors",
        nargs=3,
        type=int,
        default=[1, 2, 4],
        help="Downsample factors (e.g., 1 2 4).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        required=True,
        help="Path to the directory containing labels.",
    )
    parser.add_argument(
        "--output_hdf5", type=str, required=True, help="Path to the output HDF5 file."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Check if image and label paths exist
    assert os.path.exists(
        args.image_path
    ), f"Image path {args.image_path} does not exist."
    assert os.path.exists(
        args.label_path
    ), f"Label path {args.label_path} does not exist."

    create_hdf5(
        image_path=args.image_path,
        label_path=args.label_path,
        output_hdf5=args.output_hdf5,
        downsample_factors=tuple(args.downsample_factors),
        size=args.view_size,
    )
