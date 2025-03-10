import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, down_ratio, transform=None):
        """
        初始化数据集。

        参数：
        - h5_file_path: HDF5 文件的路径
        - transform: 图像的变换（可选）
        """
        self.h5_file_path = h5_file_path
        self.transform = transform
        # 打开 HDF5 文件（延迟打开，等到真正读取数据时再打开）
        self.h5_file = h5py.File(self.h5_file_path, "r")
        self.down_ratio = down_ratio
        # 获取样本数量
        with h5py.File(self.h5_file_path, "r") as h5_file:
            self.total_samples = len(h5_file.keys())

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        grp = self.h5_file[f"sample_{idx}"]
        # 读取数据
        ori_image = grp["image"][()]
        ori_image_wh = ori_image.shape[:2]
        pred_points = grp["pred_points"][()]
        gt_points = grp["gt_points"][()]
        is_corner = grp["is_corner"][()]
        valid_mask = grp["valid_mask"][()]
        # 转换为张量
        image = Image.fromarray(ori_image)  # PIL 图像
        pred_points = torch.from_numpy(pred_points).float()  # 形状为 (L, 2)
        gt_points = torch.from_numpy(gt_points).float()  # 形状为 (L, 2)
        is_corner = torch.from_numpy(is_corner).long()  # 形状为 (L,)

        # 获取self.transform的resize参数
        resize = None
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                resize = t.size
                break

        if resize is not None:
            # 将point位置映射到resize大小的图像上
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

            # 将point位置映射到downsample大小的图像上
            pred_points = pred_points / self.down_ratio

        # 应用图像变换
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
