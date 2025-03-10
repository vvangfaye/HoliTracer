import os
import cv2
import json
import numpy as np
import argparse
from pycocotools import mask as maskUtils
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image(image_info, image_id_to_annotations, output_dir):
    """
    处理单个图像：创建掩码并保存为PNG。

    Args:
        image_info (dict): 包含图像信息的字典。
        image_id_to_annotations (dict): 图像ID到注释的映射。
        output_dir (str): 保存掩码图像的目录。

    Returns:
        str: 处理后的掩码文件名。
    """
    image_id = image_info['id']
    filename = image_info['file_name']
    height = image_info['height']
    width = image_info['width']

    # 初始化空白掩码（单通道，值为0）
    mask = np.zeros((height, width), dtype=np.uint8)

    annotations = image_id_to_annotations.get(image_id, [])

    for annotation in annotations:
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        is_crowd = annotation.get('iscrowd', 0)

        if is_crowd:
            # 处理群体注释（使用RLE）
            rle = segmentation
            if isinstance(rle, list):
                # 如果分割信息是列表，将其转换为RLE
                rle = maskUtils.frPyObjects(rle, height, width)
            decoded_mask = maskUtils.decode(rle)
            # 如果有多个RLE对象，取逻辑或
            if isinstance(decoded_mask, list):
                decoded_mask = np.any(decoded_mask, axis=0).astype(np.uint8)
            mask = np.maximum(mask, decoded_mask * 255)
        else:
            # 处理多边形分割
            if isinstance(segmentation, list):
                # 多个多边形
                for seg in segmentation:
                    # 将列表转换为形状为(-1, 2)的numpy数组
                    polygon = np.array(seg).reshape(-1, 2).astype(np.int32)
                    # 在掩码上填充多边形，颜色为255
                    cv2.fillPoly(mask, [polygon], color=255)
            else:
                print(f"未知的分割格式，注释ID: {annotation['id']}")

    # 保存掩码为PNG文件，确保对象像素值为255，背景为0
    mask_filename = os.path.splitext(filename)[0] + '.png'
    mask_filepath = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_filepath, mask)

    return mask_filename

def main(coco_json_path, output_label_dir):
    # split = 'val'
    # Path to the COCO annotations JSON file
    # coco_json_path = f'/home/wangyu/data/vector_data/WHU_building_dataset/{split}/coco_label.json'
    # coco_json_path = f'/home/wangyu/data/vector_data/WHU_building_dataset/predict/hisup.json'
    # Directory where the mask PNG files will be saved
    # output_label_path = f'/home/wangyu/data/vector_data/WHU_building_dataset/{split}/masks'
    # output_label_path = f'/home/wangyu/data/vector_data/WHU_building_dataset/predict/masks'
    # 确保输出目录存在
    os.makedirs(output_label_dir, exist_ok=True)

    # 加载COCO注释
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # 创建一个字典，将图像ID映射到其对应的注释
    image_id_to_annotations = {}

    # 考虑label和predict的情况
    if 'annotations' not in coco:
        annotations = coco

    else:
        annotations = coco['annotations']

    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

    # COCO数据集中的图像列表
    images = coco['images']

    # 使用partial固定process_image函数的参数
    process_func = partial(process_image, image_id_to_annotations=image_id_to_annotations, output_dir=output_label_dir)

    # 确定使用的CPU核心数量
    cpu_cores = cpu_count()
    pool_size = max(cpu_cores // 2, 1)  # 使用一半的CPU核心

    print(f"使用 {pool_size} 个进程进行掩码生成。")

    # 创建一个多进程池
    with Pool(pool_size) as pool:
        # 处理所有图像
        processed_files = pool.map(process_func, images)

    print("所有掩码已成功生成。")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--coco_json_path', '-c', type=str, help='Path to the COCO annotations JSON file')
    args.add_argument('--output_label_dir', '-o', type=str, help='Directory where the mask PNG files will be saved')
    # help
    args = args.parse_args()

    main(args.coco_json_path, args.output_label_dir)
