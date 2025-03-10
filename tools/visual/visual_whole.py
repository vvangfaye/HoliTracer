import cv2
import os
import json
import numpy as np
from multiprocessing import Pool

def visualize_coco_segmentation(
    image_path: str,
    json_path: str,
    save_dir: str,
    mask_color: tuple = (44, 62, 80),
    line_color: tuple = (0, 0, 255),
    point_color: tuple = (255, 255, 0),
    mask_alpha: float = 0.5,
    line_thickness: int = 10,
    point_radius: int = 5
) -> None:
    """
    在图像上可视化COCO格式的分割标注，先收集所有分割信息，最后统一绘制。
    
    Args:
        image_path (str): 输入图像的路径
        json_path (str): COCO格式JSON文件的路径
        save_dir (str): 输出图像的保存目录
        mask_color (tuple): 填充颜色的BGR值，默认为(44, 62, 80)
        line_color (tuple): 轮廓线的BGR颜色，默认为(0, 0, 255)
        point_color (tuple): 顶点的BGR颜色，默认为(255, 255, 0)
        mask_alpha (float): 填充透明度，范围[0, 1]，默认为0.5
        line_thickness (int): 轮廓线粗细，默认为10
        point_radius (int): 顶点圆半径，默认为5
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
    image_with_segmentations = image_rgb.copy()

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

    # 存储所有分割信息
    outer_contours = []  # 外部轮廓
    inner_contours = []  # 内部轮廓
    all_points = []      # 所有顶点

    # 收集分割数据
    for annotation in target_annotations:
        segmentation = annotation["segmentation"]
        if not isinstance(segmentation, list):
            print(f"未知的segmentation格式: {segmentation}")
            continue

        # 外部轮廓
        outer_points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
        outer_contours.append(outer_points)

        # 内部轮廓
        inner_points_list = [
            np.array(seg).reshape(-1, 2).astype(np.int32) for seg in segmentation[1:]
        ]
        inner_contours.extend(inner_points_list)

        # 所有顶点
        points = np.concatenate([np.array(seg).reshape(-1, 2) for seg in segmentation])
        all_points.append(points.astype(np.int32))

    # 创建二维蒙版
    height, width = image_with_segmentations.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 统一绘制所有外部轮廓到蒙版
    if outer_contours:
        cv2.fillPoly(mask, outer_contours, 255)

    # 统一绘制所有内部空洞到蒙版
    if inner_contours:
        cv2.fillPoly(mask, inner_contours, 0)

    # 创建并应用带透明度的叠加层
    overlay = image_with_segmentations.copy()
    overlay[mask == 255] = np.array(mask_color, dtype=overlay.dtype)
    cv2.addWeighted(
        overlay,
        mask_alpha,
        image_with_segmentations,
        1 - mask_alpha,
        0,
        image_with_segmentations,
    )

    # 统一绘制所有外部轮廓线
    if outer_contours:
        cv2.polylines(
            image_with_segmentations,
            outer_contours,
            isClosed=True,
            color=line_color,
            thickness=line_thickness,
        )

    # 统一绘制所有内部轮廓线
    if inner_contours:
        cv2.polylines(
            image_with_segmentations,
            inner_contours,
            isClosed=True,
            color=line_color,
            thickness=line_thickness,
        )

    # 统一绘制所有顶点
    if all_points:
        all_points_flat = np.concatenate(all_points)
        for point in all_points_flat:
            cv2.circle(
                image_with_segmentations,
                tuple(point),
                point_radius,
                point_color,
                -1,
            )

    # 保存结果（转换回BGR格式）
    output_image = cv2.cvtColor(image_with_segmentations, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(save_dir, image_name)
    cv2.imwrite(output_path, output_image)
    print(f"已保存可视化结果至: {output_path}")

def process_image(args):
    # 解包参数并调用可视化函数
    image_path, json_path, save_dir, mask_color, line_color, point_color, mask_alpha, line_thickness, point_radius = args
    visualize_coco_segmentation(
        image_path=image_path,
        json_path=json_path,
        save_dir=save_dir,
        mask_color=mask_color,
        line_color=line_color,
        point_color=point_color,
        mask_alpha=mask_alpha,
        line_thickness=line_thickness,
        point_radius=point_radius
    )

# if __name__ == "__main__":
    # water
    # image_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/img/07.jpg"
    # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/predict/1_5_10/swin_l/swinl_upernet_vlras_180_d50_32.json"
    # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/swin_l/"
    # hisup
    # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/predict/coco_result/hisup.json"
    # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/hisup/"
    
    # visualize_coco_segmentation(
    #     image_path=image_filename,
    #     json_path=json_filename,
    #     save_dir=save_dir,
    #     mask_color=(255, 0, 0),
    #     line_color=(0, 0, 255),
    #     point_color=(255, 255, 0),
    #     mask_alpha=0.2,
    #     line_thickness=10,
    #     point_radius=7
    # )

    # # building
    # image_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/img/150000_220000.jpg"
    # # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/predict/hisup/hisup_dp_2.json"
    # # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/hisup/"
    # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/predict/1_3_6/swin_l/swinl_upernet_vlras_d25_32.json"
    # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/swin_l/"
    # image_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/img/"
    # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/predict/1_3_6/swin_l/swinl_upernet_vlras_d25_32.json"
    # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/swin_l/"
    # for image_filename in os.listdir(image_dir):
    #     if image_filename.endswith(".jpg") and not image_filename.startswith("._"):
    #         image_filename = os.path.join(image_dir, image_filename)
    #         visualize_coco_segmentation(
    #             image_path=image_filename,
    #             json_path=json_filename,
    #             save_dir=save_dir,
    #             mask_color=(255, 0, 0),
    #             line_color=(0, 0, 255),
    #             point_color=(255, 255, 0),
    #             mask_alpha=0.2,
    #             line_thickness=10,
    #             point_radius=7
    #         )

    
    


if __name__ == "__main__":
    # label
    # image_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/img/"
    # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/coco_label_with_hole_nodp.json"
    # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/label_nodp/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
    #                 if f.endswith(".jpg") and not f.startswith("._")]
    # tasks = [(image_file, json_filename, save_dir, (255, 0, 0), (0, 0, 255),
    #         (255, 255, 0), 0.2, 10, 7) for image_file in image_files]
    # num_processes = min(os.cpu_count(), len(image_files))
    # with Pool(processes=num_processes) as pool:
    #     pool.map(process_image, tasks)
        
    # our methods
    # image_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/img/"
    # json_filename = "/home/data/vector_data/VHR_road_dataset/raw/test/predict/coco_result/swinl_upernet_vlras_135_d25_32.json"
    # save_dir = "/home/data/vector_data/VHR_road_dataset/raw/test/visual/swin_l/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
    #                 if f.endswith(".jpg") and not f.startswith("._")]
    # tasks = [(image_file, json_filename, save_dir, (255, 0, 0), (0, 0, 255),
    #         (255, 255, 0), 0.2, 10, 7) for image_file in image_files]
    # num_processes = min(os.cpu_count(), len(image_files))
    # with Pool(processes=num_processes) as pool:
    #     pool.map(process_image, tasks)
    
    # compare different methods
    # methods = ["hisup", "boundaryformer", "e2ec", "fctl", "ffl", "snake", "topdig", "tsmta"]
    methods = ["ffl"]
    for method in methods:
        image_dir = f"/home/data/vector_data/VHR_road_dataset/raw/test/img/"
        json_filename = f"/home/data/vector_data/VHR_road_dataset/raw/test/predict/coco_result/{method}.json"
        save_dir = f"/home/data/vector_data/VHR_road_dataset/raw/test/visual/{method}/"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 收集所有图像文件路径
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                    if f.endswith(".jpg") and not f.startswith("._")]
        
        # 定义每个任务的参数
        tasks = [(image_file, json_filename, save_dir, (255, 0, 0), (0, 0, 255), 
                (255, 255, 0), 0.2, 10, 7) for image_file in image_files]
        
        # 创建进程池并执行任务
        num_processes = min(os.cpu_count(), len(image_files))
        with Pool(processes=num_processes) as pool:
            pool.map(process_image, tasks)