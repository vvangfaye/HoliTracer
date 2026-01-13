import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, down_ratio, transform=None):
        """
        Initialize dataset.

        Parameters:
        - h5_file_path: Path to HDF5 file
        - transform: Image transformation (optional)
        """
        self.h5_file_path = h5_file_path
        self.transform = transform
        # Open HDF5 file (lazy open, open when actually reading data)
        self.h5_file = h5py.File(self.h5_file_path, "r")
        self.down_ratio = down_ratio
        # Get number of samples
        with h5py.File(self.h5_file_path, "r") as h5_file:
            self.total_samples = len(h5_file.keys())

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        grp = self.h5_file[f"sample_{idx}"]
        # Read data
        ori_image = grp["image"][()]
        ori_image_wh = ori_image.shape[:2]
        pred_points = grp["pred_points"][()]
        gt_points = grp["gt_points"][()]
        is_corner = grp["is_corner"][()]
        valid_mask = grp["valid_mask"][()]
        # Convert to tensor
        image = Image.fromarray(ori_image)  # PIL Image
        pred_points = torch.from_numpy(pred_points).float()  # Shape is (L, 2)
        gt_points = torch.from_numpy(gt_points).float()  # Shape is (L, 2)
        is_corner = torch.from_numpy(is_corner).long()  # Shape is (L,)

        # Get resize parameter from self.transform
        resize = None
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                resize = t.size
                break

        if resize is not None:
            # Map point positions to resized image
            pred_points = (
                pred_points
                * torch.tensor(resize).float()
                / torch.tensor(image.size).float()
            )
            gt_points = (
                gt_points
                * torch.tensor(resize).float()
                / torch.tensor(image.size).float()
            )

            # Map point positions to downsampled image
            pred_points = pred_points / self.down_ratio

        # Apply image transformation
        if self.transform:
            image = self.transform(image)

        return {
            "idx": idx,
            "ori_image_wh": ori_image_wh,
            "image": image,
            "pred_points": pred_points,
            "gt_points": gt_points,
            "is_corner": is_corner,
            "valid_mask": valid_mask,
        }

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()