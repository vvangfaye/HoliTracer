import cv2
import torch
import numpy as np
from .utils import prepare_image, group_points_func, restore_group_points, interpolate_polygon_by_distance, compute_polygon_angles, counter2polygon


def vector_predict_api(
    model,
    image_path,
    polys,
    d=25,
    num_points=32,
    down_ratio=4,
    corner_threshold=0.1,
    device="cuda",
):
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        return []
    refined_annotations = []

    # 处理每一个预测
    for i, polygon in enumerate(polys):
        if polygon is None:
            # print("The segmentation is None.")
            continue

        # fliter small polygons
        if polygon.area < 100:
            continue
            
        # 需要考虑转成多个多边形的情况
        if polygon.geom_type == "MultiPolygon":
            polygons = list(polygon.geoms)
        else:
            polygons = [polygon]
        
        for poly_num, polygon in enumerate(polygons):
            
            if polygon is None:
                # print("The polygon is None.")
                continue

            # dp简化
            simple_polygon = polygon.simplify(5.0, preserve_topology=True)

            if simple_polygon is None:
                # print("The simple polygon is None.")
                continue
            try:
                refactored_polygon = interpolate_polygon_by_distance(simple_polygon, d=d)
            except Exception as e:
                # print(e)
                continue
            refactored_polygon_arrays = []
            exterior_refactored_polygon_list = refactored_polygon.exterior.coords
            refactored_polygon_array = np.array(exterior_refactored_polygon_list)[:-1]
            refactored_polygon_arrays.append(refactored_polygon_array)

            interior_refactored_polygon_lists = refactored_polygon.interiors
            if interior_refactored_polygon_lists:
                for interior_refactored_line in interior_refactored_polygon_lists:
                    interior_refactored_polygon_list = interior_refactored_line.coords
                    interior_refactored_polygon_array = np.array(
                        interior_refactored_polygon_list
                    )[:-1]
                    refactored_polygon_arrays.append(interior_refactored_polygon_array)

                simple_points_list = []
                exterior_simple_polygon_list = simple_polygon.exterior.coords
                exterior_simple_polygon_array = np.array(
                    exterior_simple_polygon_list
                ).copy()[:-1]
                exterior_simple_points = exterior_simple_polygon_array.reshape(-1, 2)
                simple_points_list.append(exterior_simple_points)

                interior_simple_polygon_lists = simple_polygon.interiors
                if interior_simple_polygon_lists:
                    for interior_simple_line in interior_simple_polygon_lists:
                        interior_simple_polygon_list = interior_simple_line.coords
                        interior_simple_polygon_array = np.array(
                            interior_simple_polygon_list
                        ).copy()[:-1]
                        interior_simple_points = interior_simple_polygon_array.reshape(
                            -1, 2
                        )
                        simple_points_list.append(interior_simple_points)

                refactored_points_list = []
                for refactored_polygon_array in refactored_polygon_arrays:
                    refactored_points = refactored_polygon_array.reshape(-1, 2)
                    refactored_points_list.append(refactored_points)

            refined_points_list = []
            refined_corner_list = []
            angle_list = []
            corner_list = []

            exterior_flag = False  # 用于标记外部多边形是否存在
            for i, refactored_polygon_array in enumerate(refactored_polygon_arrays):
                # 转换多边形点为numpy数组
                # Flatten all points into one array
                segmentation_points = refactored_polygon_array.reshape(-1, 2)

                if segmentation_points.shape[0] < 3:
                    # print("The number of all points is less than 3.")
                    continue

                # 按num_points对分割点进行分组
                point_groups, valid_counts = group_points_func(
                    segmentation_points, num_points
                )

                # 收集所有精炼后的点以便合并
                refined_all_groups = []
                corner_all_groups = []

                for group_idx, (group_points, valid_count) in enumerate(
                    zip(point_groups, valid_counts)
                ):
                    # 计算带有重叠的边界框
                    x_min = max(int(np.min(group_points[:valid_count, 0])) - 50, 0)
                    x_max = min(
                        int(np.max(group_points[:valid_count, 0])) + 50,
                        image.shape[1],
                    )
                    y_min = max(int(np.min(group_points[:valid_count, 1])) - 50, 0)
                    y_max = min(
                        int(np.max(group_points[:valid_count, 1])) + 50,
                        image.shape[0],
                    )

                    # 裁剪图像
                    image_crop = image[y_min:y_max, x_min:x_max, :]

                    # 调整分割点坐标到裁剪后的图像坐标系
                    adjusted_points = group_points.copy()
                    adjusted_points[:valid_count, 0] -= x_min
                    adjusted_points[:valid_count, 1] -= y_min

                    # 准备模型输入
                    # 将图像转换为张量并归一化
                    image_tensor = prepare_image(image_crop, device)

                    # 准备分割点张量
                    pred_points_tensor = (
                        torch.from_numpy(adjusted_points).unsqueeze(0).float()
                    )

                    # 准备valid_mask张量
                    valid_mask = torch.zeros((1, num_points), dtype=torch.float32).to(
                        device
                    )
                    valid_mask[0, :valid_count] = 1.0

                    # resize pred_points_tensor
                    pred_points_tensor = (
                        pred_points_tensor
                        * torch.tensor([512, 512]).float()
                        / torch.tensor([x_max - x_min, y_max - y_min]).float()
                        / torch.tensor(down_ratio).float()
                    )
                    pred_points_tensor = pred_points_tensor.to(device)

                    try:
                        # 运行模型
                        refined_points, is_corner_logits = model(
                            image_tensor, pred_points_tensor, valid_mask
                        )

                        refined_points = refined_points.cpu().numpy()[0]
                        is_corner_probs = torch.sigmoid(is_corner_logits).cpu().numpy()[0]

                        # 只保留有效角点
                        refined_points = refined_points[:valid_count, :]
                        is_corner_probs = is_corner_probs[:valid_count]

                        # 处理超出图像边界的点
                        refined_points[:, 0] = np.clip(refined_points[:, 0], 0, 512)
                        refined_points[:, 1] = np.clip(refined_points[:, 1], 0, 512)

                        # 调整精炼后的点回原始图像坐标系
                        refined_points[:, 0] *= (x_max - x_min) / 512
                        refined_points[:, 1] *= (y_max - y_min) / 512

                        refined_points[:, 0] += x_min
                        refined_points[:, 1] += y_min

                        # 收集精炼后的点
                        refined_all_groups.append(refined_points)
                        corner_all_groups.append(is_corner_probs)

                    except Exception as e:
                        print(e)
                        continue

                # 合并所有组的精炼点
                if refined_all_groups:
                    merged_refined_points, merged_corners, num_merged_points = (
                        restore_group_points(
                            refined_all_groups,
                            corner_all_groups,
                            num_points,
                        )
                    )
                    angles = compute_polygon_angles(merged_refined_points)

                    # # 选择角点
                    is_corner_mask = merged_corners > corner_threshold
                    #
                    # # 仅保留角点
                    refined_corner_points = merged_refined_points[is_corner_mask, :].copy()
                    # refined_corner_points = merged_refined_points.copy()
                    #
                    # 保留角度小于135度的点
                    # angles_mask = angles < (165 / 180 * math.pi)
                    # merged_refined_points = merged_refined_points[angles_mask, :]

                    if merged_refined_points.shape[0] < 3:
                        # print("The number of all merged points is less than 3.")
                        continue

                    refined_points_list.append(merged_refined_points.copy())
                    refined_corner_list.append(refined_corner_points)
                    angle_list.append(angles)
                    corner_list.append(merged_corners)

                    if i == 0:
                        exterior_flag = True

            # 如果外部多边形不存在，则跳过
            if not exterior_flag:
                # print("The merged geom exterior polygon does not exist.")
                continue

            # 计算bbox
            x_min = np.min(merged_refined_points[:, 0])
            x_max = np.max(merged_refined_points[:, 0])
            y_min = np.min(merged_refined_points[:, 1])
            y_max = np.max(merged_refined_points[:, 1])
            bbox_coco = [x_min, y_min, x_max - x_min, y_max - y_min]
            bbox_coco = [round(float(x), 2) for x in bbox_coco]

            # 将合并后的多边形转换为COCO格式的分割
            refined_segmentation = []
            outer_flag = False
            for i, refined_points in enumerate(refined_corner_list):
                segmentation = refined_points.flatten().tolist()
                # 确保多边形有至少三个点
                if len(segmentation) < 6:
                    # print("The number of all merged points is less than 3.")
                    continue
                refined_segmentation.append(segmentation)
                if i == 0:
                    outer_flag = True

            if not outer_flag:
                # print("The merged coco exterior polygon does not exist.")
                continue
            
            area = polygon.area
            
            merged_result = {
                "id": poly_num,
                "category_id": 1,
                "segmentation": refined_segmentation,
                "bbox": bbox_coco,
                "score": max(0.9, min(area / (512 * 512), 0.99))
            }
            refined_annotations.append(merged_result)
            
    return refined_annotations