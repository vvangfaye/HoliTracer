import json
import h5py
from shapely.geometry import Polygon, LineString
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from shapely.geometry.polygon import orient
import multiprocessing
from functools import partial
import math
import argparse


def load_coco_annotations(coco_file):
    """加载COCO格式的注释文件，按图像ID分组。"""
    with open(coco_file, "r") as f:
        coco_data = json.load(f)
    annotations = coco_data["annotations"]
    images = coco_data.get("images", [])
    image_info = {img["id"]: img for img in images}
    # 按照image_id分组注释
    annotations_by_image = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    print(
        f"Loaded {len(images)} images and {len(annotations)} annotations from {coco_file}"
    )
    return annotations_by_image, image_info


def coco_segmentation_to_polygons(segmentation):
    """
    将COCO格式的segmentation转换为Shapely的Polygon对象列表，考虑内部孔洞。
    """
    polygons = []
    if isinstance(segmentation, list):
        # COCO格式中，segmentation是一个列表，可能包含多个多边形（带孔洞）
        # 根据索引顺序，第一个多边形为外部轮廓，后续的为内部孔洞
        exterior = None
        interiors = []
        for idx, segment in enumerate(segmentation):
            coords = segment
            # 将坐标转换为点列表
            ring_coords = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if idx == 0:
                # 第一个多边形为外部轮廓
                exterior = ring_coords
            else:
                # 后续多边形为内部孔洞
                interiors.append(ring_coords)
        if exterior:
            polygon = Polygon(exterior, interiors)
            if polygon.is_valid:
                polygons.append(polygon)
    else:
        # 处理RLE格式或其他情况
        pass
    return polygons


def compute_iou(poly1, poly2):
    """计算两个多边形的IoU值。"""
    if not poly1.is_valid or not poly2.is_valid:
        return 0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0
    else:
        return intersection / union


def match_polygons(pred_polygons, gt_polygons, iou_threshold=0.8):
    """
    匹配预测多边形和真值多边形，返回匹配对的字典。
    """
    matches = {}
    gt_matched = set()
    for i, pred_poly in enumerate(pred_polygons):
        max_iou = 0
        matched_gt_idx = -1
        for j, gt_poly in enumerate(gt_polygons):
            if j in gt_matched:
                continue  # 已匹配的真值多边形不再考虑
            iou = compute_iou(pred_poly, gt_poly)
            if iou > max_iou:
                max_iou = iou
                matched_gt_idx = j
        if matched_gt_idx != -1 and max_iou >= iou_threshold:
            matches[i] = matched_gt_idx
            gt_matched.add(matched_gt_idx)
    return matches


def interpolate_polygon_by_distance(poly, d):
    """
    Interpolates a polygon so that no two consecutive points are more than d pixels apart.

    Parameters:
        poly (shapely.geometry.Polygon): The polygon to interpolate.
        d (float): The maximum allowed distance between consecutive points.

    Returns:
        interpolated_coords (List[Tuple[float, float]]): The interpolated coordinates.
    """
    interpolated_coords = []
    exterior = list(poly.exterior.coords)
    for i in range(len(exterior) - 1):  # exclude the repeated last point
        p1 = np.array(exterior[i])
        p2 = np.array(exterior[i + 1])
        interpolated_coords.append(tuple(p1))
        segment_length = np.linalg.norm(p2 - p1)
        if segment_length > d:
            num_extra = math.ceil(segment_length / d) - 1
            for j in range(1, num_extra + 1):
                t = j / (num_extra + 1)
                interpolated_pt = tuple(p1 + t * (p2 - p1))
                interpolated_coords.append(interpolated_pt)
    interpolated_coords.append(interpolated_coords[0])  # close the polygon
    interpolated_poly = Polygon(interpolated_coords)
    if not interpolated_poly.is_valid:
        interpolated_poly = interpolated_poly.buffer(0)
    # 确保多边形是顺时针方向
    interpolated_poly = orient(interpolated_poly, sign=-1.0)  # sign=-1.0 表示顺时针
    return interpolated_poly


def rotate_polygon_to_start_at(pred_coords, start_idx):
    """
    Rotates the polygon coordinates so that the point at start_idx becomes the first point.

    Parameters:
        pred_coords (List[Tuple[float, float]]): Coordinates of the predicted polygon.
        start_idx (int): Index of the point to start the rotation.

    Returns:
        rotated_coords (List[Tuple[float, float]]): Rotated coordinates.
    """
    return pred_coords[start_idx:] + pred_coords[:start_idx]


def modified_match_points(pred_poly, gt_poly, d):
    """
    使用线性顺序匹配的完整流程：
    1. 对Pred多边形差值。
    2. 找到与GT第一个点最接近的Pred点作为起始点
    3. 从第一个GT点开始，沿Pred点顺序匹配后续GT点，保证一对一、顺序不乱。
    4. 完成匹配后，对每对匹配点之间的GT点进行插值，保证Pred和GT之间的点数一致。
    """
    pred_poly = orient(pred_poly, sign=-1.0)  # 确保Pred多边形是顺时针方向
    gt_poly = orient(gt_poly, sign=-1.0)  # 确保GT多边形是顺时针方向

    # Step 1: 对Pred多边形插值
    interpolated_pred_poly = interpolate_polygon_by_distance(pred_poly, d)
    pred_coords = list(interpolated_pred_poly.exterior.coords)[:-1]
    gt_coords_original = list(gt_poly.exterior.coords)[:-1]

    pred_array = np.array(pred_coords)
    gt_array = np.array(gt_coords_original)

    N_pred = len(pred_coords)
    M_gt = len(gt_coords_original)
    if N_pred == 0 or M_gt == 0 or N_pred < M_gt:
        return [], [], []

    # Step 2: 找到与GT[0]最近的Pred点作为起始点
    first_gt_point = gt_array[0]
    distances_to_first_gt = np.linalg.norm(pred_array - first_gt_point, axis=1)
    closest_pred_idx = np.argmin(distances_to_first_gt)

    # 旋转Pred使closest_pred_idx为起点
    rotated_pred_coords = rotate_polygon_to_start_at(pred_coords, closest_pred_idx)
    rotated_pred_array = np.array(rotated_pred_coords)

    ordered_pred_coords = rotated_pred_coords
    ordered_pred_array = rotated_pred_array

    # Step 3: 基于顺序的线性匹配
    # GT[0] 已由起始点选择决定匹配到 Pred[0]
    matched_pred_indices = [0]
    matched_gt_indices = [0]
    is_vertex_flags = [True]

    current_pred_idx = 0

    # 依次匹配GT[1...M_gt-1]
    for gt_idx in range(1, M_gt):
        gt_pt = gt_array[gt_idx]
        if current_pred_idx + 1 >= len(ordered_pred_array):
            # 没有更多Pred点可匹配
            break

        candidate_pred_points = ordered_pred_array[current_pred_idx + 1 :]
        distances = np.linalg.norm(candidate_pred_points - gt_pt, axis=1)
        min_dist_idx = np.argmin(distances)
        chosen_pred_idx = current_pred_idx + 1 + min_dist_idx

        matched_pred_indices.append(chosen_pred_idx)
        matched_gt_indices.append(gt_idx)
        is_vertex_flags.append(True)  # 原始GT顶点

        current_pred_idx = chosen_pred_idx
        if current_pred_idx == len(ordered_pred_array) - 1:
            break

    # Step 4: 处理闭合段
    # 确保从最后一个匹配点回到第一个匹配点
    if len(matched_pred_indices) >= 2 and len(matched_gt_indices) >= 2:
        last_pred_idx = matched_pred_indices[-1]
        first_pred_idx = matched_pred_indices[0]

        if first_pred_idx != last_pred_idx:
            # 计算闭合段的剩余点数
            remaining = first_pred_idx - last_pred_idx - 1
            if remaining < 0:
                remaining += len(ordered_pred_coords)

            matched_pred_indices.append(first_pred_idx)
            matched_gt_indices.append(0)  # 闭合回第一个GT点
            is_vertex_flags.append(True)

    # 根据匹配到的索引获取坐标
    matched_gt_points = [gt_coords_original[i] for i in matched_gt_indices]

    # Step 5: 计算每对匹配点之间的剩余Pred点数
    remaining_pred_points_between = []
    num_matched = len(matched_pred_indices) - 1
    for i in range(num_matched):
        start_p = matched_pred_indices[i]
        end_p = matched_pred_indices[i + 1]
        remaining = end_p - start_p - 1
        if remaining < 0:
            remaining += len(ordered_pred_coords)
        remaining_pred_points_between.append(remaining)

    # Step 6: 对GT点进行插值
    interpolated_gt_points = []
    interpolated_is_vertex = []
    for i in range(len(matched_gt_indices) - 1):
        start_g = matched_gt_indices[i]
        end_g = matched_gt_indices[i + 1]
        num_pred_remaining = remaining_pred_points_between[i]

        if num_pred_remaining <= 0:
            continue

        # 提取GT段
        if end_g > start_g:
            gt_segment = gt_coords_original[start_g : end_g + 1]
        else:
            # 闭合段，跨越起点
            gt_segment = gt_coords_original[start_g:] + gt_coords_original[: end_g + 1]

        gt_segment_line = LineString(gt_segment)
        total_length = gt_segment_line.length
        interval = total_length / (num_pred_remaining + 1)

        for j in range(1, num_pred_remaining + 1):
            point = gt_segment_line.interpolate(j * interval)
            interpolated_gt_points.append((point.x, point.y))
            interpolated_is_vertex.append(False)

    # Step 7: 整合最终的GT点序列
    final_gt_points = []
    final_is_vertex = []
    gt_idx_pointer = 0

    for i in range(len(matched_gt_indices) - 1):
        final_gt_points.append(matched_gt_points[i])
        final_is_vertex.append(is_vertex_flags[i])

        # 添加插值点
        num_pred_remaining = remaining_pred_points_between[i]
        for j in range(num_pred_remaining):
            final_gt_points.append(interpolated_gt_points[gt_idx_pointer])
            final_is_vertex.append(interpolated_is_vertex[gt_idx_pointer])
            gt_idx_pointer += 1

    # Step 8: 确保Pred和GT的点数一致
    assert len(final_gt_points) == len(ordered_pred_coords)

    # # 计算平均距离
    # total_distance = 0
    # for i in range(len(final_gt_points)):
    #     pred_pt = ordered_pred_coords[i]
    #     gt_pt = final_gt_points[i]
    #     total_distance += np.linalg.norm(np.array(pred_pt) - np.array(gt_pt))
    # avg_distance = total_distance / len(final_gt_points)

    # if avg_distance > d:
    #     return [], [], []

    # 返回最终匹配结果
    return final_gt_points, ordered_pred_coords, final_is_vertex


def match_points(pred_poly, gt_poly, d=10.0):
    """
    Matches two polygons' points using the modified matching process.

    Parameters:
        pred_poly (shapely.geometry.Polygon): Predicted polygon.
        gt_poly (shapely.geometry.Polygon): Ground truth polygon.
        d (float): Maximum pixel distance for interpolation.

    Returns:
        matches (List[Tuple[Tuple[float, float], Tuple[float, float], bool]]):
            Each tuple contains (Predicted point, GT point, is_vertex flag).
    """
    matched_gt_points, matched_pred_points, is_vertex_flags = modified_match_points(
        pred_poly, gt_poly, d
    )

    if not matched_gt_points or not matched_pred_points:
        return []

    matches = []
    for pred_pt, gt_pt, is_vertex in zip(
        matched_pred_points, matched_gt_points, is_vertex_flags
    ):
        matches.append((pred_pt, gt_pt, is_vertex))

    return matches


def process_single_image(
    image_id, pred_annotations_by_image, gt_annotations_by_image, d=10.0
):
    """
    Processes a single image's predictions and ground truth annotations.

    Parameters:
        image_id (int): The image ID.
        pred_annotations_by_image (dict): Pred annotations grouped by image ID.
        gt_annotations_by_image (dict): GT annotations grouped by image ID.
        d (float): Maximum pixel distance for interpolation.

    Returns:
        (image_id, polygon_point_matches): Tuple containing image ID and the matched points.
    """
    print(f"Processing image with ID {image_id}")
    pred_anns = pred_annotations_by_image.get(image_id, [])
    gt_anns = gt_annotations_by_image.get(image_id, [])

    # Convert COCO segmentation to Shapely polygons
    pred_polygons = []
    for ann in pred_anns:
        polys = coco_segmentation_to_polygons(ann["segmentation"])
        pred_polygons.extend(polys)

    gt_polygons = []
    for ann in gt_anns:
        polys = coco_segmentation_to_polygons(ann["segmentation"])
        gt_polygons.extend(polys)

    # Skip if no ground truth polygons
    if not gt_polygons:
        return (image_id, None)

    # Match polygons based on IoU
    matches = match_polygons(pred_polygons, gt_polygons)
    polygon_point_matches = {}

    for pred_idx, gt_idx in matches.items():
        pred_poly = pred_polygons[pred_idx]
        gt_poly = gt_polygons[gt_idx]
        matches = match_points(pred_poly, gt_poly, d)
        if matches:
            polygon_point_matches[pred_idx] = matches

    return (image_id, polygon_point_matches)


def process_annotations(
    pred_annotations_by_image, gt_annotations_by_image, process_num=10, d=10.0
):
    """
    Processes annotations using multiprocessing.

    Parameters:
        pred_annotations_by_image (dict): Pred annotations grouped by image ID.
        gt_annotations_by_image (dict): GT annotations grouped by image ID.
        process_num (int): Number of processes to use.
        d (float): Maximum pixel distance for interpolation.

    Returns:
        all_matches (dict): All matched points across images.
    """
    all_matches = {}
    total_images = len(pred_annotations_by_image)
    image_ids = list(pred_annotations_by_image.keys())
    print(f"Total images to process: {total_images}")

    # Create a partial function with fixed parameters
    partial_process = partial(
        process_single_image,
        pred_annotations_by_image=pred_annotations_by_image,
        gt_annotations_by_image=gt_annotations_by_image,
        d=d,
    )

    # Use multiprocessing pool
    with multiprocessing.Pool(processes=process_num) as pool:
        results = pool.map(partial_process, image_ids)
    # results = []
    # for image_id in image_ids:
    #     result = partial_process(image_id)
    #     results.append(result)

    # Collect results
    processed_images = 0
    for image_id, polygon_point_matches in results:
        if polygon_point_matches is not None:
            all_matches[image_id] = polygon_point_matches
            processed_images += 1

    print(f"Processed {processed_images} images with matches")
    return all_matches


# Existing functions: interpolate_polygon, etc.


def process_matches(
    all_matches, image_info, image_dir, L, F, h5_file, d, save_dir=None
):
    """
    Constructs the training set, generates input images, predicted points, labels, etc., and saves to an HDF5 file.

    Parameters:
        all_matches (dict): Matched points across images.
        image_info (dict): Image information dictionary.
        image_dir (str): Directory containing image files.
        L (int): Sampling size.
        F (int): Sliding step.
        h5_file (h5py.File): HDF5 file object.

    Returns:
        total_samples (int): Total number of samples generated.
    """
    total_samples = 0
    total_images = len(all_matches)
    for idx_image, (image_id, matches) in enumerate(all_matches.items()):
        print(
            f"Processing matches for image {idx_image+1}/{total_images} with ID {image_id}"
        )
        # Load original image
        if image_id not in image_info:
            continue
        file_name = image_info[image_id]["file_name"]
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # For each matched polygon
        for pred_idx, match in matches.items():
            if total_samples % 100 == 0 and total_samples > 0:
                print(f"Processed {total_samples} samples")

            # Initialize empty lists
            pred_coords = []
            gt_coords = []
            is_vertexs = []

            # Traverse the matched points and extract pred_coords, gt_coords, is_vertexs
            for pred_pt, gt_pt, is_vertex in match:
                pred_coords.append(pred_pt)  # Predicted point coordinates
                gt_coords.append(gt_pt)  # Ground truth point coordinates
                is_vertexs.append(is_vertex)  # Is vertex flag

            # Generate training data
            num_points = len(pred_coords)
            for start_idx in range(0, num_points, F):
                end_idx = start_idx + L
                valid_num_points = 0
                if end_idx > num_points and num_points < L:
                    # If less than L points, 补齐
                    if start_idx != 0:
                        continue
                    pred_pts_segment = pred_coords[start_idx:] + [(0, 0)] * (
                        L - (num_points - start_idx)
                    )
                    gt_pts_segment = gt_coords[start_idx:] + [(0, 0)] * (
                        L - (num_points - start_idx)
                    )
                    is_corner_segment = is_vertexs[start_idx:] + [0] * (
                        L - (num_points - start_idx)
                    )
                    valid_num_points = num_points
                elif end_idx > num_points and num_points >= L:
                    start_idx = num_points - L
                    pred_pts_segment = pred_coords[start_idx:]
                    gt_pts_segment = gt_coords[start_idx:]
                    is_corner_segment = is_vertexs[start_idx:]
                    valid_num_points = L
                else:
                    pred_pts_segment = pred_coords[start_idx:end_idx]
                    gt_pts_segment = gt_coords[start_idx:end_idx]
                    is_corner_segment = is_vertexs[start_idx:end_idx]
                    valid_num_points = L
                # 计算平均距离
                total_distance = 0
                for i in range(len(gt_pts_segment)):
                    pred_pt = pred_pts_segment[i]
                    gt_pt = gt_pts_segment[i]
                    total_distance += np.linalg.norm(
                        np.array(pred_pt) - np.array(gt_pt)
                    )

                avg_distance = total_distance / len(gt_pts_segment)
                if avg_distance > d:
                    continue

                # Get the minimum bounding rectangle for the segment and expand by overlap
                x_coords = [pt[0] for pt in pred_pts_segment if pt != (0, 0)]
                y_coords = [pt[1] for pt in pred_pts_segment if pt != (0, 0)]
                if not x_coords or not y_coords:
                    continue  # Skip if all points are zero
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # Expand by a certain overlap, e.g., add 50 pixels
                overlap = 50
                min_x = max(int(min_x) - overlap, 0)
                max_x = min(int(max_x) + overlap, width)
                min_y = max(int(min_y) - overlap, 0)
                max_y = min(int(max_y) + overlap, height)

                gt_min_x, gt_max_x = (
                    min([pt[0] for pt in gt_pts_segment if pt != (0, 0)]),
                    max([pt[0] for pt in gt_pts_segment if pt != (0, 0)]),
                )
                gt_min_y, gt_max_y = (
                    min([pt[1] for pt in gt_pts_segment if pt != (0, 0)]),
                    max([pt[1] for pt in gt_pts_segment if pt != (0, 0)]),
                )
                if (
                    gt_min_x < min_x
                    or gt_max_x > max_x
                    or gt_min_y < min_y
                    or gt_max_y > max_y
                ):
                    continue

                # Crop the image
                cropped_image = image[min_y:max_y, min_x:max_x]

                # Map the Pred points to the cropped image coordinate system
                pred_pts_mapped = []
                for pt in pred_pts_segment:
                    mapped_pt = (
                        (pt[0] - min_x, pt[1] - min_y) if pt != (0, 0) else (0, 0)
                    )
                    pred_pts_mapped.append(mapped_pt)

                # Map the GT points to the cropped image coordinate system
                gt_pts_mapped = []
                for pt in gt_pts_segment:
                    mapped_pt = (
                        (pt[0] - min_x, pt[1] - min_y) if pt != (0, 0) else (0, 0)
                    )
                    gt_pts_mapped.append(mapped_pt)

                # Construct training data entry
                data_image = cropped_image  # Image data, shape (H, W, 3)
                data_pred_points = np.array(
                    pred_pts_mapped, dtype=np.float32
                )  # Pred points, shape (L, 2)
                data_gt_points = np.array(gt_pts_mapped, dtype=np.float32)
                data_is_corner = np.array(
                    is_corner_segment, dtype=np.int32
                )  # Corner labels, shape (L,)
                data_valid_mask = np.array(
                    [1] * valid_num_points + [0] * (L - valid_num_points),
                    dtype=np.int32,
                )  # Valid mask, shape (L,)

                # Write data to HDF5 file
                grp = h5_file.create_group(f"sample_{total_samples}")
                grp.create_dataset("image", data=data_image, compression="gzip")
                grp.create_dataset("pred_points", data=data_pred_points)
                grp.create_dataset("gt_points", data=data_gt_points)
                grp.create_dataset("is_corner", data=data_is_corner)
                grp.create_dataset("valid_mask", data=data_valid_mask)

                total_samples += 1

                if total_samples % 100 == 0:
                    # Visualization code remains unchanged
                    fig, ax = plt.subplots(figsize=(6, 6))
                    # Display the cropped image
                    ax.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                    # Plot Pred points
                    for idx, pt in enumerate(pred_pts_mapped):
                        x, y = pt
                        if x == 0 and y == 0:
                            continue  # Skip padded points
                        if is_corner_segment[idx]:
                            ax.scatter(
                                x,
                                y,
                                color="red",
                                marker="o",
                                s=10,
                                label="Corner" if idx == 0 else "",
                            )
                        else:
                            ax.scatter(
                                x,
                                y,
                                color="blue",
                                marker="o",
                                s=10,
                                label="Non-corner" if idx == 0 else "",
                            )

                        # Draw lines between consecutive points
                        if idx > 0 and pred_pts_mapped[idx - 1] != (0, 0):
                            x1, y1 = pred_pts_mapped[idx - 1]
                            ax.plot([x1, x], [y1, y], color="cyan", linewidth=0.5)

                    # Plot GT points
                    for idx, pt in enumerate(gt_pts_mapped):
                        x, y = pt
                        if x == 0 and y == 0:
                            continue
                        ax.scatter(
                            x,
                            y,
                            color="green",
                            marker="x",
                            s=10,
                            label="Corner" if idx == 0 else "",
                        )

                    # Plot matched lines
                    for idx in range(len(pred_pts_mapped)):
                        x1, y1 = pred_pts_mapped[idx]
                        x2, y2 = gt_pts_mapped[idx]
                        if x1 == 0 and y1 == 0:
                            continue
                        if x2 == 0 and y2 == 0:
                            continue
                        ax.plot([x1, x2], [y1, y2], color="magenta", linewidth=1.0)

                    # Add legend and title
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    # ax.legend(by_label.values(), by_label.keys())
                    ax.set_title("Data Item Visualization")

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir, f"{image_id}_{pred_idx}_{start_idx}_{end_idx}.png"
                    )
                    plt.savefig(save_path, bbox_inches="tight", dpi=300)
                    plt.close()
                    # ==================================

    print(f"Total samples generated: {total_samples}")

    return total_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Process COCO annotations and generate training data in HDF5 format.")
    
    # Define command-line arguments
    parser.add_argument('--dataset', type=str, default="VHR_road_dataset",
                        help="Name of the dataset (default: VHR_road_dataset)")
    parser.add_argument('--pred_coco_file', type=str,
                        help="Path to the prediction COCO JSON file")
    parser.add_argument('--gt_coco_file', type=str,
                        help="Path to the ground truth COCO JSON file")
    parser.add_argument('--image_dir', type=str, default=None,
                        help="Path to the image directory (optional, default: None)")
    parser.add_argument('-d', '--interpolation_distance', type=int, default=25,
                        help="Interpolation distance d (default: 25)")
    parser.add_argument('-L', '--sampling_size', type=int, default=32,
                        help="Sampling size L (default: 32)")
    parser.add_argument('-F', '--sliding_step', type=int, default=4,
                        help="Sliding step F (default: 4)")
    parser.add_argument('--output_file', type=str,
                        help="Path to the output HDF5 file")
    parser.add_argument('--process_num', type=int, default=40,
                        help="Number of processes for parallel processing (default: 40)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Use provided arguments or set defaults
    dataset = args.dataset
    
    # Construct default file paths if not provided
    pred_coco_file = args.pred_coco_file
    gt_coco_file = args.gt_coco_file
    image_dir = args.image_dir
    
    # If image_dir is explicitly set to None or not provided and not defaulted
    if args.image_dir == "None":
        image_dir = None
    
    # Output file path
    output_file = args.output_file

    print("Loading prediction annotations...")
    pred_annotations_by_image, pred_image_info = load_coco_annotations(pred_coco_file)
    print("Loading ground truth annotations...")
    gt_annotations_by_image, gt_image_info = load_coco_annotations(gt_coco_file)
    print("Processing annotations...")

    # Use command-line arguments
    d = args.interpolation_distance
    all_matches = process_annotations(
        pred_annotations_by_image, gt_annotations_by_image, process_num=args.process_num, d=d
    )

    # Build training dataset and save as HDF5 file
    L = args.sampling_size
    F = args.sliding_step
    h5_filename = output_file
    print(f"Saving training data to {h5_filename}...")
    with h5py.File(h5_filename, "w") as h5_file:
        total_samples = process_matches(
            all_matches, gt_image_info, image_dir, L, F, h5_file, d, f"./{dataset}"
        )

    print(f"共生成了 {total_samples} 条训练数据，并保存到 {h5_filename}")