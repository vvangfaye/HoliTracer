import os
import json
import cv2
import numpy as np
import argparse
from shapely.geometry import Polygon
from shapely.geometry import mapping
from skimage.measure import label as ski_label, regionprops
from multiprocessing import Pool
from tqdm import tqdm

def build_polygon(contours, hierarchy, index, image_height, image_width):
    """
    递归构建带空洞的 Polygon 对象。
    
    :param contours: 轮廓列表（从 cv2.findContours 获取）
    :param hierarchy: 轮廓层次结构
    :param index: 当前轮廓的索引
    :return: Polygon 对象（可能包含空洞），若失败则返回 None
    """
    if index < 0 or index >= len(contours):
        return None
    
    # 提取当前轮廓并处理 padding
    contour = contours[index]
    contour = np.array([c.reshape(-1).tolist() for c in contour])
    contour -= 1  # 减去填充（padding）
    contour = clip_by_bound(contour, image_height, image_width)  # 限制轮廓点在图像边界内
    
    if len(contour) < 3:
        return None  # 少于3个点的轮廓无法构成多边形
    
    # 获取子轮廓（空洞）
    intp = []
    child = hierarchy[0][index][2]  # 第一个子轮廓的索引
    while child != -1:
        child_poly = build_polygon(contours, hierarchy, child, image_height, image_width)
        if child_poly is not None:
            intp.append(child_poly)
        child = hierarchy[0][child][0]  # 下一个兄弟轮廓
    
    # 创建 Polygon 对象，包含外部轮廓和内部空洞
    try:
        poly = Polygon(contour, [p.exterior.coords for p in intp if p is not None])
    except Exception as e:
        print(f"构建 Polygon 失败: {e}")
        return None
    
    return poly

def process_mask_file(args):
    # 超参数
    # 查找连通区域（对象）cv2.CHAIN_APPROX_SIMPLE CHAIN_APPROX_NONE CHAIN_APPROX_TC89_KCOS
    CONTOUR_METHOD = cv2.CHAIN_APPROX_TC89_KCOS
    AREA_FLITER_VALUE = 0
    (
        mask_filename,
        masks_dir,
        image_id_map,
        image_width,
        image_height,
        simplify_value,
        category_id,
    ) = args
    annotations = []
    images = []

    # 提取图像文件名（假设掩码文件名与图像文件名一致）
    image_filename = os.path.splitext(mask_filename)[0] + ".jpg"  # 或根据实际情况调整
    # if image_filename != "07.jpg":
    #     return None

    # 获取图像 ID
    image_id = image_id_map.get(image_filename)
    if image_id is None:
        print(f"警告: 找不到图像文件 {image_filename} 的图像 ID，跳过此掩码。")
        return None  # 跳过此掩码

    # 添加图像信息（可选，如果 ground truth JSON 已经有，可以跳过）
    image_info = {
        "id": image_id,
        "file_name": image_filename,
        "width": image_width,
        "height": image_height,
    }
    images.append(image_info)

    # 读取掩码图像
    mask_path = os.path.join(masks_dir, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"警告: 无法读取掩码图像 {mask_path}，跳过此掩码。")
        return None  # 跳过此掩码

    # 确保掩码是二值图像（0 和 255）
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    label_img = ski_label(mask > 0)
    props = regionprops(label_img)

    annotation_id = 1  # 局部注释 ID

    for prop in props:
        prop_mask = np.zeros_like(mask)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(
            prop_mask, pad_width=1, mode="constant", constant_values=0
        )

        contours, hierarchy = cv2.findContours(
            padded_binary_mask, cv2.RETR_TREE, CONTOUR_METHOD
        )
        
        poly = build_polygon(contours, hierarchy, 0, image_height, image_width)
        if poly is None:
            continue
        # if len(contours) == 0:
        #     continue

        # if len(contours) > 1:
        #     for contour, h in zip(contours, hierarchy[0]):
        #         contour = np.array([c.reshape(-1).tolist() for c in contour])
        #         # 减去填充
        #         contour -= 1
        #         contour = clip_by_bound(contour, mask.shape[0], mask.shape[1])
        #         if len(contour) < 3:
        #             continue  # 忽略少于3个点的区域
        #         intp = []
        #         closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
        #         if h[3] < 0:
        #             extp = [tuple(i) for i in closed_c]
        #         else:
        #             if cv2.contourArea(closed_c.astype(int)) > 10:
        #                 intp.append([tuple(i) for i in closed_c])

        #     poly = Polygon(extp, intp)

        # else:  # len(contours) == 1
        #     contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
        #     contour -= 1
        #     contour = clip_by_bound(contour, mask.shape[0], mask.shape[1])

        #     if len(contour) < 3:
        #         continue  # 忽略少于3个点的区域
        #     closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))

        #     poly = Polygon(closed_c)

        # 多边形简化
        simplify = True  # 是否简化多边形
        tolerance = simplify_value
        if simplify:
            poly = poly.simplify(tolerance=tolerance, preserve_topology=True)

        # 处理简化成多个多边形的情况
        if isinstance(poly, Polygon):
            # 获取分割信息（按 COCO 格式）
            p_area = round(poly.area, 2)
            # 过滤
            if p_area > AREA_FLITER_VALUE:
                p_bbox = [
                    poly.bounds[0],
                    poly.bounds[1],
                    poly.bounds[2] - poly.bounds[0],
                    poly.bounds[3] - poly.bounds[1],
                ]
                # 过滤
                if p_bbox[2] > 5 and p_bbox[3] > 5:
                    p_seg = []
                    coor_list = mapping(poly)["coordinates"]
                    for part_poly in coor_list:
                        p_seg.append(np.asarray(part_poly).ravel().tolist())
                    anno_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "segmentation": p_seg,
                        "area": p_area,
                        "bbox": p_bbox,
                        "category_id": category_id,
                        "iscrowd": 0,
                        "score": max(0.9, min(p_area / (512 * 512), 0.99))
                    }
                    annotations.append(anno_info)
                    annotation_id += 1
        else:
            for idx in range(len(poly.geoms)):
                p = poly.geoms[idx]
                p_area = round(p.area, 2)
                if p_area > AREA_FLITER_VALUE:
                    p_bbox = [
                        p.bounds[0],
                        p.bounds[1],
                        p.bounds[2] - p.bounds[0],
                        p.bounds[3] - p.bounds[1],
                    ]
                    if p_bbox[2] > 5 and p_bbox[3] > 5:
                        p_seg = []
                        coor_list = mapping(p)["coordinates"]
                        for part_poly in coor_list:
                            p_seg.append(np.asarray(part_poly).ravel().tolist())
                        anno_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "segmentation": p_seg,
                            "area": p_area,
                            "bbox": p_bbox,
                            "category_id": category_id,
                            "iscrowd": 0,
                            "score": max(0.9, min(p_area / (512 * 512), 0.99))
                        }
                        annotations.append(anno_info)
                        annotation_id += 1

    return images, annotations


def masks_to_coco_predictions(
    masks_dir,
    save_json_path,
    image_id_map,
    image_width,
    image_height,
    simplify_value,
    num_processes=1,
):
    """
    将二值掩码图像转换为 COCO 格式的预测 JSON 文件（多进程版本）。

    Args:
        masks_dir (str): 掩码图像所在的目录。
        save_json_path (str): 保存生成的 COCO JSON 文件的路径。
        image_id_map (dict): 图像文件名到图像 ID 的映射。
        image_width (int): 图像的宽度（像素）。
        image_height (int): 图像的高度（像素）。
    """
    from functools import partial

    coco_dict = {"images": [], "annotations": [], "categories": []}

    # 定义类别（假设只有一个类别）
    category_id = 1
    category_name = "building"  # 根据实际情况替换类别名称
    coco_dict["categories"].append(
        {"id": category_id, "name": category_name, "supercategory": "none"}
    )

    # 获取所有掩码图像文件
    mask_files = [
        f for f in os.listdir(masks_dir) if f.endswith(".png") and not f.startswith(".")
    ]

    # 准备参数列表
    args_list = [
        (
            mask_filename,
            masks_dir,
            image_id_map,
            image_width,
            image_height,
            simplify_value,
            category_id,
        )
        for mask_filename in mask_files
    ]

    # 使用多进程处理
    with Pool(num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_mask_file, args_list), total=len(args_list), ncols=100
            )
        )

    annotation_id = 1  # 全局注释 ID

    for result in results:
        if result is None:
            continue
        images, annotations = result
        for image_info in images:
            if image_info not in coco_dict["images"]:
                coco_dict["images"].append(image_info)
        for anno_info in annotations:
            anno_info["id"] = annotation_id
            coco_dict["annotations"].append(anno_info)
            annotation_id += 1

    # 保存 COCO JSON 文件
    with open(save_json_path, "w") as json_file:
        json.dump(coco_dict, json_file, indent=4)

    print(f"预测的 COCO 格式 JSON 文件已保存至：{save_json_path}")


# 辅助函数
def clip_by_bound(contour, height, width):
    contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
    return contour


# 示例使用
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--masks_directory",
        type=str,
        default="predict/hisup_old",
        help="The directory of predict mask images.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="predict/hisup.json",
        help="The path to save the generated COCO JSON file.",
    )
    parser.add_argument(
        "--ground_truth_json",
        type=str,
        default="test/coco_label.json",
        help="The path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--simplify_value",
        type=float,
        default=3,
        help="The tolerance value for simplifying polygons.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="whubuilding",
        help="The dataset name.",
    )
    parser.add_argument(
        "--num_prcess",
        "-n",
        type=int,
        default=1,
        help="Number of processes to use for multiprocessing.",
    )

    args = parser.parse_args()
    # args.masks_directory = "/home/wangyu/data/vector_data/WHU_building_dataset/predict/hisup_old"  # 预测的掩码图像所在的目录
    # args.output_json = "/home/wangyu/data/vector_data/WHU_building_dataset/predict/hisup.json"  # 保存生成的预测 COCO JSON 文件的路径

    # 加载 ground truth JSON 以获取图像文件名到图像 ID 的映射
    # args.ground_truth_json = "/home/wangyu/data2/WHU_building_dataset/test/coco_label_with_inter.json"   # 替换为实际路径
    with open(args.ground_truth_json, "r") as f:
        ground_truth = json.load(f)

    image_id_map = {image["file_name"]: image["id"] for image in ground_truth["images"]}

    img_width = img_height = 0
    if args.dataset == "whubuilding":
        img_width = img_height = 10000
    elif args.dataset == "glhwater":
        img_width = img_height = 12800
    elif args.dataset == "vhrroad":
        img_width = img_height = 12500

    save_dir = os.path.dirname(args.output_json)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    masks_to_coco_predictions(
        args.masks_directory,
        args.output_json,
        image_id_map,
        img_width,
        img_height,
        args.simplify_value,
        args.num_prcess,
    )
