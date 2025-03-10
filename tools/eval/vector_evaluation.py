import numpy as np
import json
import argparse
import cv2
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as maskUtils
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from shapely import geometry
from shapely.geometry import Polygon
from multiprocessing import Pool

def bounding_box(points):
    """returns a list containing the bottom left and the top right
    points in the sequence
    Here, we traverse the collection of points only once,
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float("inf"), float("inf")
    top_right_x, top_right_y = float("-inf"), float("-inf")
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]


def compare_polys(poly_a, poly_b):
    """Compares two polygons via the "polis" distance metric.
    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.
    Input:
        poly_a: A Shapely polygon.
        poly_b: Another Shapely polygon.
    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    dist = polis(bndry_a.coords, bndry_b)
    dist += polis(bndry_b.coords, bndry_a)
    return dist


def polis(coords, bndry):
    """Computes one side of the "polis" metric.
    Input:
        coords: A Shapley coordinate sequence (presumably the vertices
                of a polygon).
        bndry: A Shapely linestring (presumably the boundary of
        another polygon).

    Returns:
        The "polis" metric for this pair.  You usually compute this in
        both directions to preserve symmetry.
    """
    sum = 0.0
    for pt in (
        geometry.Point(c) for c in coords[:-1]
    ):  # Skip the last point (same as first)
        sum += bndry.distance(pt)
    return sum / float(2 * len(coords))


def evaluate_img_polis(args):
    imgId, gts, dts = args

    if len(gts) == 0 or len(dts) == 0:
        return 0

    gt_bboxs = [
        bounding_box(np.array(gt["segmentation"][0]).reshape(-1, 2)) for gt in gts
    ]
    dt_bboxs = [
        bounding_box(np.array(dt["segmentation"][0]).reshape(-1, 2)) for dt in dts
    ]
    gt_polygons = [np.array(gt["segmentation"][0]).reshape(-1, 2) for gt in gts]
    dt_polygons = [np.array(dt["segmentation"][0]).reshape(-1, 2) for dt in dts]

    # IoU match
    iscrowd = [0] * len(gt_bboxs)
    ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

    # compute polis
    img_polis_avg = 0
    num_sample = 0
    min_match_num = 0
    for i, gt_poly in enumerate(gt_polygons):
        matched_idx = np.argmax(ious[:, i])
        iou = ious[matched_idx, i]
        if iou > 0.5:
            polis_value = compare_polys(
                Polygon(gt_poly), Polygon(dt_polygons[matched_idx])
            )
            img_polis_avg += polis_value
            num_sample += 1
            min_match_num += 1

    if num_sample == 0:
        return 0
    else:
        return img_polis_avg / num_sample

class PolisEval:
    def __init__(self, cocoGt=None, cocoDt=None):
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.stats = []
        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))

    def _prepare(self):
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"]].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self, num_processes):
        self._prepare()

        per_image_data = []
        for imgId in self.imgIds:
            gts = self._gts[imgId]
            dts = self._dts[imgId]
            per_image_data.append((imgId, gts, dts))

        with Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(evaluate_img_polis, per_image_data),
                    total=len(per_image_data),
                )
            )

        polis_tot = sum(res for res in results if res != 0)
        num_valid_imgs = sum(1 for res in results if res != 0)

        polis_avg = polis_tot / num_valid_imgs if num_valid_imgs > 0 else 0

        print("average polis: %f" % (polis_avg))

        return polis_avg


def polis_eval(annFile, resFile, num_processes):
    print("=" * 20 + "Polis Evaluation" + "=" * 20)
    # print(annFile)
    # print(resFile)
    gt_coco = COCO(annFile)
    if isinstance(resFile, str):
        pred_json = json.loads(open(resFile).read())["annotations"]
    else:
        pred_json = resFile["annotations"]
    dt_coco = gt_coco.loadRes(pred_json)
    polisEval = PolisEval(gt_coco, dt_coco)
    polis = polisEval.evaluate(num_processes=num_processes)
    print("=" * 50)
    return polis


def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I / (U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou


def calc_f1(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    i = np.logical_and(a, b)
    I = np.sum(i)
    A = np.sum(a)
    B = np.sum(b)

    f1 = 2 * I / (A + B + 1e-9)

    is_void = (A + B) == 0
    if is_void:
        return 1.0
    else:
        return f1

# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
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
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def calc_BoundaryIoU(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt = gt.astype(np.uint8)
    dt = dt.astype(np.uint8)
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def compute_iou_ciou_for_image(args):
    img, gt_annotations, dt_annotations = args
    img_height = img["height"]
    img_width = img["width"]
    N_GT = 0
    if len(gt_annotations) > 0:
        for _idx, annotation in enumerate(gt_annotations):
            rle = cocomask.frPyObjects(
                annotation["segmentation"], img_height, img_width
            )
            m = cocomask.decode(rle)
            if m.shape[2] > 1:
                m = m[:, :, 0]
            if _idx == 0:
                mask_gt = m.reshape((img_height, img_width))
                N_GT = len(annotation["segmentation"][0]) // 2
            else:
                mask_gt = mask_gt + m.reshape((img_height, img_width))
                N_GT = N_GT + len(annotation["segmentation"][0]) // 2
    else:
        mask_gt = np.zeros((img_height, img_width), dtype=np.uint8)
    mask_gt = mask_gt != 0

    N = 0
    if len(dt_annotations) > 0:
        for _idx, annotation in enumerate(dt_annotations):
            rle = cocomask.frPyObjects(
                annotation["segmentation"], img_height, img_width
            )
            m = cocomask.decode(rle)
            if m.shape[2] > 1:
                m = m[:, :, 0]
            if _idx == 0:
                mask = m.reshape((img_height, img_width))
                N = len(annotation["segmentation"][0]) // 2
            else:
                mask = mask + m.reshape((img_height, img_width))
                N = N + len(annotation["segmentation"][0]) // 2
    else:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
    mask = mask != 0
    # print('N_GT:', N_GT, 'N:', N)
    ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
    iou = calc_IoU(mask, mask_gt)
    f1 = calc_f1(mask, mask_gt)

    biou = calc_BoundaryIoU(mask, mask_gt)
    ciou = iou * ps

    return f1, iou, ciou, biou, ps


def compute_IoU_cIoU_bIoU(input_json, gti_annotations, num_processes):
    print("=" * 20 + "F1, IoU and CIoU Evaluation" + "=" * 20)
    # Ground truth annotations
    coco_gt = COCO(gti_annotations)

    # load predicted annotations
    if isinstance(input_json, str):
        pred_json = json.loads(open(input_json).read())["annotations"]
    else:
        pred_json = input_json["annotations"]
    coco_dt = coco_gt.loadRes(pred_json)

    image_ids = coco_gt.getImgIds(catIds=coco_gt.getCatIds())

    per_image_data = []
    for image_id in image_ids:
        img = coco_gt.loadImgs(image_id)[0]

        # get GT annotations for this image
        annotation_ids = coco_gt.getAnnIds(imgIds=img["id"])
        gt_annotations = coco_gt.loadAnns(annotation_ids)

        # get DT annotations for this image
        annotation_ids = coco_dt.getAnnIds(imgIds=img["id"])
        dt_annotations = coco_dt.loadAnns(annotation_ids)

        per_image_data.append((img, gt_annotations, dt_annotations))

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(compute_iou_ciou_for_image, per_image_data),
                total=len(per_image_data),
            )
        )

    list_f1 = [res[0] for res in results]
    list_iou = [res[1] for res in results]
    list_ciou = [res[2] for res in results]
    list_biou = [res[3] for res in results]
    pss = [res[4] for res in results]

    print("Done!")
    print("Mean F1: ", np.mean(list_f1))
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))
    print("Mean BIoU: ", np.mean(list_biou))
    print("=" * 50)

    return np.mean(list_f1), np.mean(list_iou), np.mean(list_ciou), np.mean(list_biou)


def coco_eval(annFile, resFile, num_processes):
    print(20 * "-" + "COCO evaluation" + 20 * "-")
    type = 1
    annType = ["bbox", "segm"]
    print("Running demo for *%s* results." % (annType[type]))

    cocoGt = COCO(annFile)
    if isinstance(resFile, str):
        with open(resFile) as f:
            pre_json = json.load(f)["annotations"]
    else:
        pre_json = resFile["annotations"]
    cocoDt = cocoGt.loadRes(pre_json)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]

    cocoEval.params.maxDets = [500, 1000, 2000]
    cocoEval.params.areaRng = [
        [0**2, 1e5**2],
        [0**2, 128**2],
        [128**2, 512**2],
        [512**2, 1e5**2],
    ]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats