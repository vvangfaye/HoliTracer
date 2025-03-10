import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_polygon_angles, compute_polygon_angles_with_loop


class NormalLoss(nn.Module):
    def __init__(self, weight_regression=1.0, weight_classification=1.0):
        super(NormalLoss, self).__init__()
        self.weight_regression = weight_regression
        self.weight_classification = weight_classification
        self.poly_loss = nn.SmoothL1Loss()  # 平滑 L1 损失
        self.classification_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, displacements_pred, displacements_gt, is_corner_logits, is_corner_gt
    ):
        """
        displacements_pred: (B, num_points, 2)
        displacements_gt: (B, num_points, 2)
        is_corner_logits: (B, num_points)
        is_corner_gt: (B, num_points)
        """
        # 位移差值损失（回归）
        regression_loss = self.poly_loss(displacements_pred, displacements_gt)
        # 角点分类损失（分类）
        classification_loss = self.classification_loss(
            is_corner_logits, is_corner_gt.float()
        )
        # 总损失
        total_loss = (
            regression_loss * self.weight_regression
            + classification_loss * self.weight_classification
        )
        return total_loss, regression_loss.item(), classification_loss.item()


class NormalLossWithMask(nn.Module):
    def __init__(self, weight_regression=1.0, weight_classification=1.0):
        super(NormalLossWithMask, self).__init__()
        self.weight_regression = weight_regression
        self.weight_classification = weight_classification
        self.poly_loss = nn.SmoothL1Loss(reduction="none")  # 设置为'none'以便应用掩码
        self.classification_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        displacements_pred,
        displacements_gt,
        is_corner_logits,
        is_corner_gt,
        valid_mask,
    ):
        """
        Args:
            displacements_pred: (B, num_points, 2)
            displacements_gt: (B, num_points, 2)
            is_corner_logits: (B, num_points)
            is_corner_gt: (B, num_points)
            valid_mask: (B, num_points) - Int tensor with 1 for valid points
        Returns:
            total_loss: Combined weighted loss
            regression_loss: Mean regression loss for valid points
            classification_loss: Mean classification loss for valid points
        """
        # 将int掩码转换为bool并扩展维度用于回归损失
        valid_mask_bool = valid_mask.bool()
        valid_mask_regression = valid_mask_bool.unsqueeze(-1).expand_as(
            displacements_pred
        )

        # 计算回归损失（对每个点的每个坐标计算损失）
        point_regression_losses = self.poly_loss(displacements_pred, displacements_gt)
        valid_regression_losses = point_regression_losses[valid_mask_regression]
        regression_loss = valid_regression_losses.mean()

        # 计算分类损失
        point_classification_losses = self.classification_loss(
            is_corner_logits, is_corner_gt.float()
        )
        valid_classification_losses = point_classification_losses[valid_mask_bool]
        classification_loss = valid_classification_losses.mean()

        # 计算总损失
        total_loss = (
            regression_loss * self.weight_regression
            + classification_loss * self.weight_classification
        )

        return total_loss, regression_loss.item(), classification_loss.item()


class AngleLossWithMask(nn.Module):
    def __init__(
        self,
        weight_regression=1.0,
        weight_classification=1.0,
        weight_angle=1.0,  # 角度损失的权重，默认为0表示关闭
        corner_threshold=30,
        noncorner_threshold=150,
    ):
        super(AngleLossWithMask, self).__init__()
        self.weight_regression = weight_regression
        self.weight_classification = weight_classification
        self.weight_angle = weight_angle

        self.corner_thresh = corner_threshold / 180.0 * 3.1415926
        self.noncorner_thresh = noncorner_threshold / 180.0 * 3.1415926

        # 原有损失
        self.poly_loss = nn.SmoothL1Loss(reduction="none")
        self.classification_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        displacements_pred,  # (B, N, 2)
        displacements_gt,  # (B, N, 2)
        is_corner_logits,  # (B, N)   - 用于BCE分类
        is_corner_gt,  # (B, N)   - 1/0表示是否角点
        valid_mask,  # (B, N)
    ):
        """
        Args:
            displacements_pred: (B, num_points, 2)  [若是多边形顶点的绝对坐标或回归结果]
            displacements_gt:   (B, num_points, 2)
            is_corner_logits:   (B, num_points)
            is_corner_gt:       (B, num_points)
            valid_mask:         (B, num_points) - 1表示有效点，0表示无效
        Returns:
            total_loss: Combined weighted loss
            regression_loss: Mean regression loss for valid points
            classification_loss: Mean classification loss for valid points
            angle_loss: Mean angle loss for valid points (若 weight_angle=0则为0)
        """

        # (1) 回归损失
        valid_mask_bool = valid_mask.bool()
        valid_mask_regression = valid_mask_bool.unsqueeze(-1).expand_as(
            displacements_pred
        )

        point_regression_losses = self.poly_loss(displacements_pred, displacements_gt)
        valid_regression_losses = point_regression_losses[valid_mask_regression]
        regression_loss = valid_regression_losses.mean()

        # (2) 分类损失
        point_classification_losses = self.classification_loss(
            is_corner_logits, is_corner_gt.float()
        )
        valid_classification_losses = point_classification_losses[valid_mask_bool]
        classification_loss = valid_classification_losses.mean()

        # (3) 角度损失（若 weight_angle>0 则计算，否则为0）
        angle_loss = 0.0
        if self.weight_angle > 0.0:
            # 假设 displacements_pred 就是多边形顶点的绝对坐标
            with torch.no_grad():
                angles_pred = compute_polygon_angles_with_loop(
                    displacements_pred, valid_mask
                )  # (B, N)

            # margin-based angle loss
            # label=1 => 想让 angle <= corner_thresh =>  penalty = max(0, angle - corner_thresh)
            # label=0 => 想让 angle >= noncorner_thresh => penalty = max(0, noncorner_thresh - angle)
            corner_mask = is_corner_gt.float()  # (B, N)
            corner_part = corner_mask * F.relu(angles_pred - self.corner_thresh)
            noncorner_part = (1 - corner_mask) * F.relu(
                self.noncorner_thresh - angles_pred
            )

            angle_loss_tensor = corner_part + noncorner_part
            # 只对有效点计算
            angle_loss_tensor = angle_loss_tensor[valid_mask_bool]
            angle_loss = angle_loss_tensor.mean()

        # (4) 总损失
        total_loss = (
            regression_loss * self.weight_regression
            + classification_loss * self.weight_classification
            + angle_loss * self.weight_angle
        )

        # 返回损失
        return (
            total_loss,
            regression_loss.item(),
            classification_loss.item(),
            float(angle_loss),
        )
