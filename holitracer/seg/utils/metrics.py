import os, cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import warnings

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

class Evaluator(object):
    def __init__(self, num_class, ignore_index=-1, device='cpu'):
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self):
        # 使用 PyTorch 张量存储混淆矩阵，方便设备间通信
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class), dtype=torch.float64, device=self.device)

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc.item()

    def Pixel_Accuracy_Class(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        Acc = torch.nanmean(Acc)
        return Acc.item()

    def Mean_Intersection_over_Union(self):
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) - intersection
        IoU = intersection / union
        MIoU = torch.nanmean(IoU)
        return MIoU.item()
    
    def Intersection_over_Union(self):
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) - intersection
        IoU = intersection / union
        return IoU.cpu().numpy()

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, dim=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU.item()

    def F1_Score(self, idx=-1):
        precision = torch.diag(self.confusion_matrix) / torch.sum(self.confusion_matrix, dim=0)
        recall = torch.diag(self.confusion_matrix) / torch.sum(self.confusion_matrix, dim=1)
        f1_score = 2 * precision * recall / (precision + recall)
        if idx < 0:
            return torch.nanmean(f1_score).item()
        else:
            return f1_score[idx].item()

    def generate_matrix(self, gt_image, pre_image):
        mask = gt_image != self.ignore_index
        gt = gt_image[mask]
        pred = pre_image[mask]
        n = self.num_class
        # 确保标签在合理范围内
        k = (gt >= 0) & (gt < n)
        inds = n * gt[k].to(torch.int64) + pred[k].to(torch.int64)
        confusion_matrix = torch.bincount(inds, minlength=n**2).reshape(n, n).to(torch.float64)
        return confusion_matrix.to(self.device)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, f"{gt_image.shape} != {pre_image.shape}"
        batch_confusion = self.generate_matrix(gt_image.to(self.device), pre_image.to(self.device))
        self.confusion_matrix += batch_confusion

    def synchronize_between_processes(self):
        """
        使用 torch.distributed.all_reduce 来汇总所有进程的混淆矩阵
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.confusion_matrix, op=torch.distributed.ReduceOp.SUM)

    def cal_metrics(self):
        precision = torch.diag(self.confusion_matrix) / torch.sum(self.confusion_matrix, dim=0)
        recall = torch.diag(self.confusion_matrix) / torch.sum(self.confusion_matrix, dim=1)
        f1_score = 2 * precision * recall / (precision + recall)
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) - intersection
        iou = intersection / union
        oa = intersection.sum() / self.confusion_matrix.sum()

        print("Precision:\n", precision.cpu().numpy())
        print("Recall:\n", recall.cpu().numpy())
        print("F1 Score:\n", f1_score.cpu().numpy())
        print("IoU:\n", iou.cpu().numpy())

        print("mPrecision:\n", torch.nanmean(precision).item())
        print("mRecall:\n", torch.nanmean(recall).item())
        print("mF1:\n", torch.nanmean(f1_score).item())
        print("mIoU:\n", torch.nanmean(iou).item())
        print("FWIoU:\n", self.Frequency_Weighted_Intersection_over_Union())

        print("Overall Accuracy (OA):\n", oa.item())


def cal_metrics(pre_path, ann_path, nclass, ignore_index):
    print("====================evaluating====================")
    print(f"pre_path = {pre_path}")
    print(f"ann_path = {ann_path}")
    print(f"ncalss = {nclass}")
    print(f"ignore_index = {ignore_index}")
    evaluator = Evaluator(nclass, ignore_index=ignore_index)
    evaluator.reset()
    for name in tqdm(os.listdir(pre_path)):
        ann = np.array(Image.open(os.path.join(ann_path, name))).astype(np.uint8)
        pre = np.array(Image.open(os.path.join(pre_path, name))).astype(np.uint8)
        if nclass == 2:
            ann[ann > 0] = 1
            pre[pre > 0] = 1
        evaluator.add_batch(ann, pre)
    evaluator.cal_metrics()
    print("==================================================")


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.2):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.2):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    # boundary_iou = intersection / union
    return intersection, union  # , boundary_iou


def cal_boundaryIoU(pre_path, ann_path, dilation_ratio=0.2):
    print("====================boundaryIoU====================")
    print(f"pre_path = {pre_path}")
    print(f"ann_path = {ann_path}")
    print(f"dilation_ratio={dilation_ratio}")
    boundary_intersection = 0
    boundary_union = 0
    for name in tqdm(os.listdir(labels)):
        pre = np.array(Image.open(os.path.join(pres, name))).astype("uint8")
        ann = np.array(Image.open(os.path.join(labels, name))).astype("uint8")
        pre[pre < 128] = 0
        pre[pre >= 128] = 1
        ann[ann < 100] = 0
        ann[ann > 100] = 1
        i, u = boundary_iou(gt=ann, dt=pre, dilation_ratio=dilation_ratio)
        boundary_union += u
        boundary_intersection += i
    print('boundary_iou: %.8f%%' % (boundary_intersection / boundary_union * 100))
    print("===================================================")


if __name__ == '__main__':
    # cal_metrics(pre_path=r"D:\git\wbuilding\PFNet_IPNet\result\512\upernet\context_upernet512_1_5_10",
    #             ann_path=r"F:\data_lwc\FBP\test\label",
    #             nclass=25,
    #             ignore_index=0)
    labels = rf"F:\data_lwc\FBP\test\label"
    out_path_m = rf"F:\data_lwc\FBP\test\ipnet"
    cal_metrics(pre_path=out_path_m, ann_path=labels, nclass=25, ignore_index=0)
    print(xxx)
    cal_metrics(pre_path=r"F:\data_lwc\HPD\result\HPD_normal_swinl_upernet_512_1",
                ann_path=r"F:\data_lwc\HPD\test\label",
                nclass=10,
                ignore_index=0)
    cal_metrics(pre_path=r"F:\data_lwc\HPD\result\HPD_context_swinl_upernet_512_1_5_10",
                ann_path=r"F:\data_lwc\HPD\test\label",
                nclass=10,
                ignore_index=0)

