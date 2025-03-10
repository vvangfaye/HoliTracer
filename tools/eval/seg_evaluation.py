import numpy as np
import argparse
from collections import Counter
from tqdm import tqdm
import os
from PIL import Image
from multiprocessing import Pool
import cv2  # Added cv2 for boundary IoU computation

# Ignore the warning: image too big
Image.MAX_IMAGE_PIXELS = None


def get_confusion_matrix(label, pred, class_num=2, ignore=False):
    confu_list = []
    h = 1 if ignore else 0
    for i in range(h, class_num):
        c = Counter(pred[np.where(label == i)])
        single_row = []
        for j in range(h, class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.uint64)


def get_metrics(confusion_matrix_total):
    class_num = confusion_matrix_total.shape[0]

    confu_mat = confusion_matrix_total.astype(np.float64) + 1e-18
    col_sum = np.sum(confu_mat, axis=1)  # Sum over rows
    raw_sum = np.sum(confu_mat, axis=0)  # Sum over columns

    oa = np.trace(confu_mat) / np.sum(confu_mat)

    TP = np.diag(confu_mat)
    FN = col_sum - TP
    FP = raw_sum - TP

    miou = np.diag(confu_mat) / (col_sum + raw_sum - np.diag(confu_mat))
    miou = np.nanmean(miou)

    precision = TP[1] / (TP[1] + FP[1]) if (TP[1] + FP[1]) != 0 else 0
    recall = TP[1] / (TP[1] + FN[1]) if (TP[1] + FN[1]) != 0 else 0
    iou = TP[1] / (TP[1] + FP[1] + FN[1]) if (TP[1] + FP[1] + FN[1]) != 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    metrics = {
        "Overall_Accuracy": round(oa, 9),
        "Precision": round(precision, 9),
        "Recall": round(recall, 9),
        "Iou": round(iou, 9),
        "MIou": round(miou, 9),
        "F1_score": round(f1_score, 9),
    }

    return metrics


def mask_to_boundary(mask, dilation_ratio=5):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation_ratio)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=5):
    """
    Compute boundary IoU between two binary masks.
    :param gt (numpy array, uint8): ground truth binary mask
    :param dt (numpy array, uint8): predicted binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary intersection and union counts
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    return intersection, union


def process_single_image(args):
    name, pre_path, ann_path = args
    label_path = os.path.join(ann_path, name)
    pred_path = os.path.join(pre_path, name)
    label = np.array(Image.open(label_path)).astype(np.uint8)
    pred = np.array(Image.open(pred_path)).astype(np.uint8)
    # Thresholding the labels and predictions
    label[label < 100] = 0
    label[label >= 100] = 1
    # pred[pred < 127] = 0
    pred[pred > 0] = 1

    confusion_matrix = get_confusion_matrix(pred=pred, label=label)
    boundary_intersection, boundary_union = boundary_iou(
        gt=label, dt=pred, dilation_ratio=5
    )

    return confusion_matrix, boundary_intersection, boundary_union


def Get_metrics(ann_path, pre_path, num_process):
    print(f"pre_path = {pre_path}")
    print(f"ann_path = {ann_path}")
    print("Computing metrics...")

    image_names = os.listdir(ann_path)
    image_names = [
        name
        for name in image_names
        if name.endswith(".png") and not name.startswith(".")
    ]
    args_list = [(name, pre_path, ann_path) for name in image_names]

    with Pool(num_process) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_single_image, args_list),
                total=len(args_list),
            )
        )
    # results = []
    # for args in args_list:
    #     result = process_single_image(args)
    #     results.append(result)

    confu_matrices = []
    boundary_intersections = []
    boundary_unions = []

    for confu_matrix, boundary_intersection, boundary_union in results:
        confu_matrices.append(confu_matrix)
        boundary_intersections.append(boundary_intersection)
        boundary_unions.append(boundary_union)

    confu_matrix_total = np.sum(confu_matrices, axis=0)
    total_boundary_intersection = np.sum(boundary_intersections)
    total_boundary_union = np.sum(boundary_unions)

    metrics = get_metrics(confu_matrix_total)
    boundary_iou_value = (
        total_boundary_intersection / total_boundary_union
        if total_boundary_union != 0
        else 0
    )
    print(boundary_iou_value)
    metrics["Boundary_IoU"] = round(boundary_iou_value, 9)

    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation metrics including Boundary IoU."
    )
    parser.add_argument(
        "--gt_path",
        "-g",
        type=str,
        default="/home/data/vector_data/WHU_building_dataset/raw/test/mask/",
        # required=True,
        help="The path of ground truth masks.",
    )
    parser.add_argument(
        "--pred_path",
        "-p",
        type=str,
        default="/home/data/vector_data/WHU_building_dataset/raw/test/predict/swin_l/",
        # required=True,
        help="The path of predicted masks.",
    )
    parser.add_argument(
        "--num_process",
        "-n",
        type=int,
        default=8,
        help="The number of processes to use.",
    )
    args = parser.parse_args()

    Get_metrics(args.gt_path, args.pred_path, args.num_process)
