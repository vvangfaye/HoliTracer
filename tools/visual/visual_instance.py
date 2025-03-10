import cv2
import os
import json
import numpy as np

def visualize_coco_segmentation(
    image_path: str,
    json_path: str,
    save_dir: str,
    top_n: int = 3,
    mask_color: tuple = (44, 62, 80),
    line_color: tuple = (0, 0, 255),
    point_color: tuple = (255, 255, 0),
    mask_alpha: float = 0.5,
    line_thickness: int = 10,
    point_radius: int = 5,
    buffer_size: int = 50
) -> None:
    """
    在图像上可视化COCO格式的分割标注，绘制面积前n大的实例，并分别保存。
    
    Args:
        image_path (str): 输入图像的路径
        json_path (str): COCO格式JSON文件的路径
        save_dir (str): 输出图像的保存目录
        top_n (int): 要绘制的最大面积实例数量，默认为3
        mask_color (tuple): 填充颜色的BGR值
        line_color (tuple): 轮廓线的BGR颜色
        point_color (tuple): 顶点的BGR颜色
        mask_alpha (float): 填充透明度，范围[0, 1]
        line_thickness (int): 轮廓线粗细
        point_radius (int): 顶点圆半径
        buffer_size (int): 裁剪缓冲区大小，默认为50像素
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载COCO JSON文件
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # 获取目标图像ID
    image_name = os.path.basename(image_path)
    target_image_id = None
    for img in coco_data["images"]:
        if img["file_name"][:-4] == image_name[:-4]:
            target_image_id = img["id"]
            break
    if target_image_id is None:
        print(f"未在JSON中找到匹配的图像: {image_name}")
        return

    # 筛选目标图像的annotation
    target_annotations = [
        anno for anno in coco_data["annotations"] if anno["image_id"] == target_image_id
    ]
    if not target_annotations:
        print(f"未找到图像 {image_name} 的分割标注")
        return

    # 计算每个实例的面积并排序
    annotations_with_area = []
    for annotation in target_annotations:
        if not isinstance(annotation["segmentation"], list):
            continue
        outer_points = np.array(annotation["segmentation"][0]).reshape(-1, 2)
        area = cv2.contourArea(outer_points.astype(np.float32))
        annotations_with_area.append((area, annotation))
    
    # 按面积从大到小排序并取前n个
    annotations_with_area.sort(reverse=True)
    top_annotations = annotations_with_area[:min(top_n, len(annotations_with_area))]

    if not top_annotations:
        print(f"未找到有效的分割数据: {image_name}")
        return

    # 对每个前n大的实例进行处理
    for rank, (area, annotation) in enumerate(top_annotations, 1):
        image_with_segmentations = image_rgb.copy()
        
        # 处理当前实例
        segmentation = annotation["segmentation"]
        outer_points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
        inner_points_list = [
            np.array(seg).reshape(-1, 2).astype(np.int32) for seg in segmentation[1:]
        ]
        all_points = np.concatenate([np.array(seg).reshape(-1, 2) for seg in segmentation]).astype(np.int32)

        # 计算边界框并添加缓冲区
        x, y, w, h = cv2.boundingRect(outer_points)
        x_min = max(0, x - buffer_size)
        y_min = max(0, y - buffer_size)
        x_max = min(image_rgb.shape[1], x + w + buffer_size)
        y_max = min(image_rgb.shape[0], y + h + buffer_size)

        # 裁剪图像
        image_cropped = image_with_segmentations[y_min:y_max, x_min:x_max].copy()

        # 创建蒙版（相对于裁剪后的图像）
        height, width = image_cropped.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # 调整坐标到裁剪后的图像坐标系
        outer_points_cropped = outer_points - [x_min, y_min]
        inner_points_cropped = [points - [x_min, y_min] for points in inner_points_list]
        all_points_cropped = all_points - [x_min, y_min]

        # 绘制蒙版
        cv2.fillPoly(mask, [outer_points_cropped], 255)
        if inner_points_cropped:
            cv2.fillPoly(mask, inner_points_cropped, 0)

        # 应用带透明度的叠加层
        overlay = image_cropped.copy()
        overlay[mask == 255] = np.array(mask_color, dtype=overlay.dtype)
        cv2.addWeighted(
            overlay,
            mask_alpha,
            image_cropped,
            1 - mask_alpha,
            0,
            image_cropped,
        )

        # 绘制轮廓线
        cv2.polylines(
            image_cropped,
            [outer_points_cropped],
            isClosed=True,
            color=line_color,
            thickness=line_thickness,
        )
        if inner_points_cropped:
            cv2.polylines(
                image_cropped,
                inner_points_cropped,
                isClosed=True,
                color=line_color,
                thickness=line_thickness,
            )

        # 绘制顶点
        for point in all_points_cropped:
            cv2.circle(
                image_cropped,
                tuple(point),
                point_radius,
                point_color,
                -1,
            )

        # 保存结果
        output_image = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(save_dir, f"top{rank}_{image_name}")
        cv2.imwrite(output_path, output_image)
        print(f"已保存第{rank}大实例可视化结果至: {output_path} (面积: {area})")

if __name__ == "__main__":
    image_filename = "/home/data/vector_data/WHU_building_dataset/raw/test/img/150000_220000.jpg"
    # json_filename = "/home/data/vector_data/WHU_building_dataset/raw/test/predict/hisup/hisup_dp_2.json"
    # save_dir = "/home/data/vector_data/WHU_building_dataset/raw/test/visual/hisup/"
    json_filename = "/home/data/vector_data/WHU_building_dataset/raw/test/predict/1_3_6/swin_l/swinl_upernet_vlras_d25_32.json"
    save_dir = "/home/data/vector_data/WHU_building_dataset/raw/test/visual/swin_l/"
    visualize_coco_segmentation(
        image_path=image_filename,
        json_path=json_filename,
        save_dir=save_dir,
        top_n=10,  # 可修改为需要的数量
        mask_color=(255, 0, 0),
        line_color=(0, 0, 255),
        point_color=(255, 255, 0),
        mask_alpha=0.2,
        line_thickness=10,
        point_radius=7,
        buffer_size=50
    )