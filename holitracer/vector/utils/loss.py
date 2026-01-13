import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_polygon_angles, compute_polygon_angles_with_loop


class NormalLoss(nn.Module):
    def __init__(self, weight_regression=1.0, weight_classification=1.0):
        super(NormalLoss, self).__init__()
        self.weight_regression = weight_regression
        self.weight_classification = weight_classification
        self.poly_loss = nn.SmoothL1Loss()  # Smooth L1 Loss
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
        # Displacement difference loss (Regression)
        regression_loss = self.poly_loss(displacements_pred, displacements_gt)
        # Corner classification loss (Classification)
        classification_loss = self.classification_loss(
            is_corner_logits, is_corner_gt.float()
        )
        # Total loss
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
        self.poly_loss = nn.SmoothL1Loss(reduction="none")  # Set to 'none' to apply mask
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
        # Convert int mask to bool and expand dimension for regression loss
        valid_mask_bool = valid_mask.bool()
        valid_mask_regression = valid_mask_bool.unsqueeze(-1).expand_as(
            displacements_pred
        )

        # Calculate regression loss (loss for each coordinate of each point)
        point_regression_losses = self.poly_loss(displacements_pred, displacements_gt)
        valid_regression_losses = point_regression_losses[valid_mask_regression]
        regression_loss = valid_regression_losses.mean()

        # Calculate classification loss
        point_classification_losses = self.classification_loss(
            is_corner_logits, is_corner_gt.float()
        )
        valid_classification_losses = point_classification_losses[valid_mask_bool]
        classification_loss = valid_classification_losses.mean()

        # Calculate total loss
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
        weight_angle=1.0,  # Weight of angle loss, default 0 means disabled
        corner_threshold=30,
        noncorner_threshold=150,
    ):
        super(AngleLossWithMask, self).__init__()
        self.weight_regression = weight_regression
        self.weight_classification = weight_classification
        self.weight_angle = weight_angle

        self.corner_thresh = corner_threshold / 180.0 * 3.1415926
        self.noncorner_thresh = noncorner_threshold / 180.0 * 3.1415926

        # Original loss
        self.poly_loss = nn.SmoothL1Loss(reduction="none")
        self.classification_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        displacements_pred,  # (B, N, 2)
        displacements_gt,  # (B, N, 2)
        is_corner_logits,  # (B, N)   - For BCE classification
        is_corner_gt,  # (B, N)   - 1/0 indicates if it is a corner
        valid_mask,  # (B, N)
    ):
        """
        Args:
            displacements_pred: (B, num_points, 2)  [If absolute coordinates of polygon vertices or regression results]
            displacements_gt:   (B, num_points, 2)
            is_corner_logits:   (B, num_points)
            is_corner_gt:       (B, num_points)
            valid_mask:         (B, num_points) - 1 indicates valid point, 0 indicates invalid
        Returns:
            total_loss: Combined weighted loss
            regression_loss: Mean regression loss for valid points
            classification_loss: Mean classification loss for valid points
            angle_loss: Mean angle loss for valid points (0 if weight_angle=0)
        """

        # (1) Regression loss
        valid_mask_bool = valid_mask.bool()
        valid_mask_regression = valid_mask_bool.unsqueeze(-1).expand_as(
            displacements_pred
        )

        point_regression_losses = self.poly_loss(displacements_pred, displacements_gt)
        valid_regression_losses = point_regression_losses[valid_mask_regression]
        regression_loss = valid_regression_losses.mean()

        # (2) Classification loss
        point_classification_losses = self.classification_loss(
            is_corner_logits, is_corner_gt.float()
        )
        valid_classification_losses = point_classification_losses[valid_mask_bool]
        classification_loss = valid_classification_losses.mean()

        # (3) Angle loss (calculated if weight_angle > 0, otherwise 0)
        angle_loss = 0.0
        if self.weight_angle > 0.0:
            # Assuming displacements_pred are the absolute coordinates of polygon vertices
            with torch.no_grad():
                angles_pred = compute_polygon_angles_with_loop(
                    displacements_pred, valid_mask
                )  # (B, N)

            # margin-based angle loss
            # label=1 => Want angle <= corner_thresh =>  penalty = max(0, angle - corner_thresh)
            # label=0 => Want angle >= noncorner_thresh => penalty = max(0, noncorner_thresh - angle)
            corner_mask = is_corner_gt.float()  # (B, N)
            corner_part = corner_mask * F.relu(angles_pred - self.corner_thresh)
            noncorner_part = (1 - corner_mask) * F.relu(
                self.noncorner_thresh - angles_pred
            )

            angle_loss_tensor = corner_part + noncorner_part
            # Only calculate for valid points
            angle_loss_tensor = angle_loss_tensor[valid_mask_bool]
            angle_loss = angle_loss_tensor.mean()

        # (4) Total loss
        total_loss = (
            regression_loss * self.weight_regression
            + classification_loss * self.weight_classification
            + angle_loss * self.weight_angle
        )

        # Return loss
        return (
            total_loss,
            regression_loss.item(),
            classification_loss.item(),
            float(angle_loss),
        )