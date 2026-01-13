import math
import os
import cv2
import torch
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import orient


def infer_visual_one_whole_image(
    model,
    image_folder,
    image_name,
    image_id,
    predictions,
    num_points,
    down_ratio,
    corner_threshold,
    d,
    device,
    visual_save_dir=None,
    visual=False,
    labels=None,
):
    """
    Perform inference and visualization of each instance on a whole image, returning the refined prediction results. (Considering internal holes of polygons)

    Args:
        model: torch.nn.Module, Model
        image_folder: str, Image folder
        image_name: str, Image filename
        image_id: int, Image ID (for COCO format)
        predictions: dict, Prediction results
        labels: dict, Label results
        num_points: int, Number of points
        down_ratio: int, Downsampling ratio
        corner_threshold: float, Corner threshold
        device: torch.device, Device

    Returns:
        list of dict, Refined prediction results
    """
    if visual_save_dir is not None:
        os.makedirs(visual_save_dir, exist_ok=True)
    # Load image
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    pred_annotations = predictions.copy()

    matches = {}
    if visual:
        gt_annotations = labels.copy()
        # match polygons for visualization
        pred_polygons = []
        for ann in pred_annotations:
            polys = coco_segmentation_to_polygons(ann["segmentation"])
            pred_polygons.append(polys)

        gt_polygons = []
        for ann in gt_annotations:
            polys = coco_segmentation_to_polygons(ann["segmentation"])
            gt_polygons.append(polys)

        matches = match_polygons(pred_polygons, gt_polygons, iou_threshold=0.8)

    if not predictions:
        return []

    refined_annotations = []

    # Process each prediction
    for ann_idx, ann in enumerate(predictions):
        # Get segmentation points
        segmentation = ann.get("segmentation")

        if segmentation is None:
            # print("The segmentation is None.")
            continue

        # Process segmentation polygon
        # Assume 'segmentation' is a list of multiple polygons, each polygon is [x1, y1, x2, y2, ..., xn, yn]
        polygon = coco_segmentation_to_polygons(segmentation)

        if polygon.area < 100:
            continue

        # Need to consider converting to multiple polygons
        if polygon.geom_type == "MultiPolygon":
            polygons = list(polygon.geoms)
        else:
            polygons = [polygon]

        for poly_num, polygon in enumerate(polygons):

            if polygon is None:
                # print("The polygon is None.")
                continue

            # dp simplification
            simple_polygon = polygon.simplify(5.0, preserve_topology=True)

            if simple_polygon is None:
                # print("The simple polygon is None.")
                continue
            try:
                refactored_polygon = interpolate_polygon_by_distance(
                    simple_polygon, d=d
                )
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

            if visual:
                ori_points_list = []
                exterior_ori_polygon_list = polygon.exterior.coords
                exterior_ori_polygon_array = np.array(exterior_ori_polygon_list).copy()[
                    :-1
                ]
                exterior_ori_points = exterior_ori_polygon_array.reshape(-1, 2)
                ori_points_list.append(exterior_ori_points)

                interior_ori_polygon_lists = polygon.interiors
                if interior_ori_polygon_lists:
                    for interior_ori_line in interior_ori_polygon_lists:
                        interior_ori_polygon_list = interior_ori_line.coords
                        interior_ori_polygon_array = np.array(
                            interior_ori_polygon_list
                        ).copy()[:-1]
                        try:
                            interior_ori_points = interior_ori_polygon_array.reshape(
                                -1, 2
                            )
                        except Exception as e:
                            print(e)
                            continue
                        ori_points_list.append(interior_ori_points)

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

            exterior_flag = False  # To mark if exterior polygon exists
            for i, refactored_polygon_array in enumerate(refactored_polygon_arrays):
                # Convert polygon points to numpy array
                # Flatten all points into one array
                segmentation_points = refactored_polygon_array.reshape(-1, 2)

                if segmentation_points.shape[0] < 3:
                    # print("The number of all points is less than 3.")
                    continue

                # Group segmentation points by num_points
                point_groups, valid_counts = group_points_func(
                    segmentation_points, num_points
                )

                # Collect all refined points for merging
                refined_all_groups = []
                corner_all_groups = []

                for group_idx, (group_points, valid_count) in enumerate(
                    zip(point_groups, valid_counts)
                ):
                    # Calculate bounding box with overlap
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

                    # Crop image
                    image_crop = image[y_min:y_max, x_min:x_max, :]

                    # Adjust segmentation point coordinates to cropped image coordinate system
                    adjusted_points = group_points.copy()
                    adjusted_points[:valid_count, 0] -= x_min
                    adjusted_points[:valid_count, 1] -= y_min

                    # Prepare model input
                    # Convert image to tensor and normalize
                    image_tensor = prepare_image(image_crop, device)

                    # Prepare segmentation points tensor
                    pred_points_tensor = (
                        torch.from_numpy(adjusted_points).unsqueeze(0).float()
                    )

                    # Prepare valid_mask tensor
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
                        # Run model
                        refined_points, is_corner_logits = model(
                            image_tensor, pred_points_tensor, valid_mask
                        )

                        refined_points = refined_points.cpu().numpy()[0]
                        is_corner_probs = (
                            torch.sigmoid(is_corner_logits).cpu().numpy()[0]
                        )

                        # Keep only valid corner points
                        refined_points = refined_points[:valid_count, :]
                        is_corner_probs = is_corner_probs[:valid_count]

                        # Handle points out of image boundaries
                        refined_points[:, 0] = np.clip(refined_points[:, 0], 0, 512)
                        refined_points[:, 1] = np.clip(refined_points[:, 1], 0, 512)

                        # Adjust refined points back to original image coordinate system
                        refined_points[:, 0] *= (x_max - x_min) / 512
                        refined_points[:, 1] *= (y_max - y_min) / 512

                        refined_points[:, 0] += x_min
                        refined_points[:, 1] += y_min

                        # Collect refined points
                        refined_all_groups.append(refined_points)
                        corner_all_groups.append(is_corner_probs)

                    except Exception as e:
                        print(e)
                        continue

                # Merge all groups of refined points
                if refined_all_groups:
                    merged_refined_points, merged_corners, num_merged_points = (
                        restore_group_points(
                            refined_all_groups,
                            corner_all_groups,
                            num_points,
                        )
                    )
                    angles = compute_polygon_angles(merged_refined_points)

                    # # Select corner points
                    is_corner_mask = merged_corners > corner_threshold
                    #
                    # # Keep only corner points
                    refined_corner_points = merged_refined_points[
                        is_corner_mask, :
                    ].copy()
                    # refined_corner_points = merged_refined_points.copy()
                    #
                    # Keep points with angle less than 135 degrees
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

            # If exterior polygon does not exist, skip
            if not exterior_flag:
                # print("The merged geom exterior polygon does not exist.")
                continue

            if visual and ann_idx in matches:
                # Save visualization results
                exterior_refined_points = refined_points_list[0].copy()

                crop_x_min = max(int(np.min(exterior_refined_points[:, 0])) - 50, 0)
                crop_x_max = min(
                    int(np.max(exterior_refined_points[:, 0])) + 50, image.shape[1]
                )
                crop_y_min = max(int(np.min(exterior_refined_points[:, 1])) - 50, 0)
                crop_y_max = min(
                    int(np.max(exterior_refined_points[:, 1])) + 50, image.shape[0]
                )

                # Save visualization results
                visual_image_crop = image[
                    crop_y_min:crop_y_max, crop_x_min:crop_x_max, :
                ]
                gt_seg = gt_annotations[matches[ann_idx]].get("segmentation")
                gt_points_trans_list = []
                for seg in gt_seg:
                    gt_points = np.array(seg).reshape(-1, 2)
                    gt_points[:, 0] -= crop_x_min
                    gt_points[:, 1] -= crop_y_min
                    gt_points_trans_list.append(gt_points)

                ori_points_trans_list = []
                for ori_points in ori_points_list:
                    ori_points = ori_points.copy()
                    ori_points[:, 0] -= crop_x_min
                    ori_points[:, 1] -= crop_y_min
                    ori_points_trans_list.append(ori_points)

                simple_points_trans_list = []
                for simple_points in simple_points_list:
                    simple_points = simple_points.copy()
                    simple_points[:, 0] -= crop_x_min
                    simple_points[:, 1] -= crop_y_min
                    simple_points_trans_list.append(simple_points)

                refactored_points_trans_list = []
                for refactored_points in refactored_points_list:
                    refactored_points = refactored_points.copy()
                    refactored_points[:, 0] -= crop_x_min
                    refactored_points[:, 1] -= crop_y_min
                    refactored_points_trans_list.append(refactored_points)

                refined_points_trans_list = []
                for refined_points in refined_points_list:
                    refined_points = refined_points.copy()
                    refined_points[:, 0] -= crop_x_min
                    refined_points[:, 1] -= crop_y_min
                    refined_points_trans_list.append(refined_points)

                refined_corner_trans_list = []
                for refined_corner in refined_corner_list:
                    refined_corner = refined_corner.copy()
                    refined_corner[:, 0] -= crop_x_min
                    refined_corner[:, 1] -= crop_y_min
                    refined_corner_trans_list.append(refined_corner)
                try:
                    visualize(
                        os.path.join(
                            visual_save_dir, image_name[:-4], str(matches[ann_idx])
                        ),
                        visual_image_crop,
                        gt_points_trans_list,
                        ori_points_trans_list,
                        simple_points_trans_list,
                        refactored_points_trans_list,
                        refined_points_trans_list,
                        corner_list,
                        angle_list,
                        corner_threshold,
                        angles_threshold=165 / 180 * math.pi,
                    )
                except Exception as e:
                    pass
                # visualize_as_video(
                #     os.path.join(visual_save_dir, image_name[:-4], str(matches[ann_idx])),
                #     visual_image_crop,
                #     ori_points_trans_list,
                #     simple_points_trans_list,
                #     refactored_points_trans_list,
                #     refined_points_trans_list,
                #     corner_list,
                #     corner_threshold,
                #     fps=3
                # )

            # Calculate bbox
            x_min = np.min(merged_refined_points[:, 0])
            x_max = np.max(merged_refined_points[:, 0])
            y_min = np.min(merged_refined_points[:, 1])
            y_max = np.max(merged_refined_points[:, 1])
            bbox_coco = [x_min, y_min, x_max - x_min, y_max - y_min]
            bbox_coco = [round(float(x), 2) for x in bbox_coco]

            # Convert merged polygon to COCO format segmentation
            refined_segmentation = []
            outer_flag = False
            for i, refined_points in enumerate(refined_corner_list):
                segmentation = refined_points.flatten().tolist()
                # Ensure polygon has at least 3 points
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

            # Assign a unique ID to each polygon
            refined_idx = ann.get("id", ann_idx) + poly_num * 1000000

            merged_result = {
                "image_id": image_id,
                "id": ann.get("id", refined_idx),
                "category_id": ann.get("category_id", 1),
                "segmentation": refined_segmentation,
                "bbox": bbox_coco,
                "score": max(0.9, min(area / (512 * 512), 0.99)),
            }
            refined_annotations.append(merged_result)
    return refined_annotations


def group_points_func(points, group_size):
    """Groups points into groups of specified size and returns the number of valid points in each group.

    Args:
        points: numpy array of shape (N, 2)
        group_size: int, Number of points per group

    Returns:
        list of numpy arrays, each of shape (group_size, 2)
        list of ints, Number of valid points per group
    """
    groups = []
    valid_counts = []
    num_points = points.shape[0]
    step = group_size // 2

    for i in range(0, num_points, step):
        start_idx = i
        end_idx = i + group_size
        if end_idx > num_points:
            group = points[start_idx:]
            valid_count = group.shape[0]
            padding = np.zeros((group_size - valid_count, 2), dtype=np.float32)
            group = np.vstack([group, padding])

            groups.append(group)
            valid_counts.append(valid_count)
            break
        else:
            group = points[start_idx:end_idx]
            valid_count = group.shape[0]

        groups.append(group)
        valid_counts.append(valid_count)

    return groups, valid_counts


def restore_group_points(groups, corner_groups, group_size):
    """Restores grouped points into a single array.

    Args:
        groups: list of numpy arrays, each of shape (group_size, 2)
        corner_groups: list of numpy arrays, each of shape (group_size,)
        group_size: int, Number of points per group

    Returns:
        numpy array of shape (N, 2)
    """
    points_num = 0
    step = group_size // 2
    restored_points = np.zeros((0, 2), dtype=np.float32)
    restored_corners = np.zeros((0,), dtype=np.float32)
    for idx, (group, corners) in enumerate(zip(groups, corner_groups)):
        if idx == 0:
            restored_points = group
            restored_corners = corners
            points_num += group.shape[0]
        else:
            overlap = group[:step]
            corners_overlap = corners[:step]

            # Calculate average of overlapping parts
            overlap_avg = np.mean([restored_points[-step:], overlap], axis=0)
            restored_points = np.vstack(
                [restored_points[:-step], overlap_avg, group[step:]]
            )

            corners_overlap_avg = np.mean(
                [restored_corners[-step:], corners_overlap], axis=0
            )
            restored_corners = np.hstack(
                [restored_corners[:-step], corners_overlap_avg, corners[step:]]
            )

            points_num += group.shape[0] - step

    return restored_points, restored_corners, points_num


def coco_segmentation_to_polygons(segmentation):
    """
    Convert COCO format segmentation to Shapely Polygon object list, considering internal holes.
    """
    if isinstance(segmentation, list):
        # In COCO format, segmentation is a list that may contain multiple polygons (with holes)
        # According to index order, the first polygon is the outer contour, and subsequent ones are internal holes
        exterior = None
        interiors = []
        for idx, segment in enumerate(segmentation):
            coords = segment
            # Convert coordinates to a list of points
            ring_coords = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if idx == 0:
                # The first polygon is the outer contour
                exterior = ring_coords
            else:
                # Subsequent polygons are internal holes
                interiors.append(ring_coords)
        if exterior:
            polygon = Polygon(exterior, interiors)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
        else:
            polygon = None
    else:
        polygon = None

    return polygon


def interpolate_polygon_by_distance(poly, d):
    """
    Interpolates a polygon so that no two consecutive points are more than d pixels apart.

    Parameters:
        poly (shapely.geometry.Polygon): The polygon to interpolate.
        d (float): The maximum allowed distance between consecutive points.

    Returns:
        interpolated_coords (List[Tuple[float, float]]): The interpolated coordinates.
    """
    interpolated_coords_exterior = []
    exterior = list(poly.exterior.coords)
    for i in range(len(exterior) - 1):  # exclude the repeated last point
        p1 = np.array(exterior[i])
        p2 = np.array(exterior[i + 1])
        interpolated_coords_exterior.append(tuple(p1))
        segment_length = np.linalg.norm(p2 - p1)
        if segment_length > d:
            num_extra = math.ceil(segment_length / d) - 1
            for j in range(1, num_extra + 1):
                t = j / (num_extra + 1)
                interpolated_pt = tuple(p1 + t * (p2 - p1))
                interpolated_coords_exterior.append(interpolated_pt)
    interpolated_coords_exterior.append(
        interpolated_coords_exterior[0]
    )  # close the polygon

    # Interpolation of internal frames
    interiors = poly.interiors
    interpolated_coords_interiors = None

    if interiors:
        interpolated_coords_interiors = []
        for interior in interiors:
            interpolated_coords_interior = []
            interior_coords = list(interior.coords)
            for i in range(len(interior_coords) - 1):
                p1 = np.array(interior_coords[i])
                p2 = np.array(interior_coords[i + 1])
                interpolated_coords_interior.append(tuple(p1))
                segment_length = np.linalg.norm(p2 - p1)
                if segment_length > d:
                    num_extra = math.ceil(segment_length / d) - 1
                    for j in range(1, num_extra + 1):
                        t = j / (num_extra + 1)
                        interpolated_pt = tuple(p1 + t * (p2 - p1))
                        interpolated_coords_interior.append(interpolated_pt)

            interpolated_coords_interior.append(
                interior_coords[0]
            )  # close the interior

            interpolated_coords_interiors.append(interpolated_coords_interior)

    interpolated_poly = Polygon(
        interpolated_coords_exterior, interpolated_coords_interiors
    )
    if not interpolated_poly.is_valid:
        interpolated_poly = interpolated_poly.buffer(0)
    # Ensure polygon is clockwise
    interpolated_poly = orient(interpolated_poly, sign=-1.0)  # sign=-1.0 indicates clockwise
    return interpolated_poly


def prepare_image(image, device):
    """Converts and normalizes the image for model input.

    Args:
        image: The input image in BGR format.

    Returns:
        A torch.Tensor of shape (1, 3, H, W).
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize the image 512
    image_rgb = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Convert to float and normalize to [0, 1]
    image_rgb = image_rgb.astype(np.float32) / 255.0

    # If your model requires mean and std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_rgb - mean) / std

    # Convert to channel-first format
    image_transposed = np.transpose(image_normalized, (2, 0, 1))

    # Convert to torch.Tensor
    image_tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(device).float()

    return image_tensor


def compute_iou(poly1, poly2):
    """Calculate IoU value of two polygons."""
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
    Match predicted polygons and ground truth polygons, returning a dictionary of matched pairs.
    """
    matches = {}
    gt_matched = set()
    for i, pred_poly in enumerate(pred_polygons):
        if pred_poly is None:
            continue
        max_iou = 0
        matched_gt_idx = -1
        for j, gt_poly in enumerate(gt_polygons):
            if j in gt_matched or gt_poly is None:
                continue  # Matched ground truth polygons are no longer considered
            iou = compute_iou(pred_poly, gt_poly)
            if iou > max_iou:
                max_iou = iou
                matched_gt_idx = j
        if matched_gt_idx != -1 and max_iou >= iou_threshold:
            matches[i] = matched_gt_idx
            gt_matched.add(matched_gt_idx)
    return matches


def draw_points(image, points, point_color, line_color, size=2):
    """Draws points on the image.

    Args:
        image: np.ndarray, The input image.
        points: np.ndarray, The points to draw.
        point_color: tuple, The color of the points.
        line_color: tuple, The color of the lines.
        size: int, The size of the points.

    Returns:
        np.ndarray, The image with points
    """
    for i in range(len(points)):
        cv2.circle(
            image,
            (
                int(points[i][0]),
                int(points[i][1]),
            ),
            size,
            point_color,
            -1,
        )
    # draw the lines between points
    for i in range(len(points) - 1):
        cv2.line(
            image,
            (
                int(points[i][0]),
                int(points[i][1]),
            ),
            (
                int(points[i + 1][0]),
                int(points[i + 1][1]),
            ),
            line_color,
            1,
        )
    # draw the line between the first and the last points
    cv2.line(
        image,
        (
            int(points[0][0]),
            int(points[0][1]),
        ),
        (
            int(points[-1][0]),
            int(points[-1][1]),
        ),
        line_color,
        1,
    )
    return image


def visualize(
    save_dir,
    image,
    gt_points_list,
    ori_points_list,
    simple_points_list,
    refactored_points_list,
    refined_points_list,
    is_corner_probs_list,
    angles_list,
    corner_threshold=0.5,
    angles_threshold=135,
):
    """Visualizes the input image and the points.

    Args:
        save_dir: str, The directory to save the visualizations.
        image: np.ndarray, The input image.
        gt_points: np.ndarray, The ground truth points.
        ori_points: np.ndarray, The original points.
        refactored_points: np.ndarray, The refactored points.
        refined_points: np.ndarray, The refined points.
        is_corner_probs: np.ndarray, The corner probabilities.
    """
    image_save_path = os.path.join(save_dir, "image.jpg")
    gt_points_image_save_path = os.path.join(save_dir, "image_gt_points.jpg")
    ori_points_image_save_path = os.path.join(save_dir, "image_ori_points.jpg")
    simple_points_image_save_path = os.path.join(save_dir, "image_simple_points.jpg")
    refactored_points_image_save_path = os.path.join(
        save_dir, "image_refactored_points.jpg"
    )
    refined_points_image_save_path = os.path.join(save_dir, "image_refined_points.jpg")
    refined_points_corner_image_save_path = os.path.join(
        save_dir, "image_refined_points_corner.jpg"
    )
    angle_image_save_path = os.path.join(save_dir, "image_angle_points.jpg")

    gt_points_save_path = os.path.join(save_dir, "point_gt_points.npy")
    ori_points_save_path = os.path.join(save_dir, "point_ori_points.npy")
    simple_points_save_path = os.path.join(save_dir, "point_simple_points.npy")
    refactored_points_save_path = os.path.join(save_dir, "point_refactored_points.npy")
    refined_points_save_path = os.path.join(save_dir, "point_refined_points.npy")
    is_corner_probs_save_path = os.path.join(save_dir, "point_is_corner_probs.npy")
    angles_save_path = os.path.join(save_dir, "point_angles.npy")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_save_path, image_array)

    color_map = {
        "gt": (0, 255, 0),
        "ori": (0, 255, 0),
        "simple": (0, 255, 0),
        "refactored": (0, 255, 0),
        "refined": (0, 255, 0),
        "refined_corner": (0, 255, 0),
        "angle": (0, 255, 0),
    }
    # beatiful lines color, not white

    gt_image_array = image_array.copy()
    i = 0
    for gt_points in gt_points_list:
        i += 1
        gt_image_array = draw_points(
            gt_image_array, gt_points, color_map["gt"], color_map["gt"], size=3
        )
        # if i > 1:
        #     print(gt_points_image_save_path + " has inerior holes")
    cv2.imwrite(gt_points_image_save_path, gt_image_array)

    ori_image_array = image_array.copy()
    for ori_points in ori_points_list:
        ori_image_array = draw_points(
            ori_image_array, ori_points, color_map["ori"], color_map["ori"], size=3
        )
    cv2.imwrite(ori_points_image_save_path, ori_image_array)

    simple_image_array = image_array.copy()
    for simple_points in simple_points_list:
        simple_image_array = draw_points(
            simple_image_array,
            simple_points,
            color_map["simple"],
            color_map["simple"],
            size=3,
        )
    cv2.imwrite(simple_points_image_save_path, simple_image_array)

    refactored_image_array = image_array.copy()
    for refactored_points in refactored_points_list:
        refactored_image_array = draw_points(
            refactored_image_array,
            refactored_points,
            color_map["refactored"],
            color_map["refactored"],
            size=3,
        )
    cv2.imwrite(refactored_points_image_save_path, refactored_image_array)

    refined_image_array = image_array.copy()
    for refined_points, is_corner_probs in zip(
        refined_points_list, is_corner_probs_list
    ):
        refined_image_array = draw_points(
            refined_image_array,
            refined_points,
            color_map["refined"],
            color_map["refined"],
            size=3,
        )
    cv2.imwrite(refined_points_image_save_path, refined_image_array)

    refined_corner_image_array = image_array.copy()
    for refined_points, is_corner_probs in zip(
        refined_points_list, is_corner_probs_list
    ):
        refined_corner_image_array = draw_points(
            refined_corner_image_array,
            refined_points[is_corner_probs > corner_threshold],
            color_map["refined_corner"],
            color_map["refined_corner"],
            size=3,
        )
    cv2.imwrite(refined_points_corner_image_save_path, refined_corner_image_array)

    # draw probs heatmap with refactored points coordinates and is_corner_probs on the image size array
    background = np.zeros_like(image_array)
    for refactored_points, is_corner_probs in zip(
        refactored_points_list, is_corner_probs_list
    ):
        for point, prob in zip(refactored_points, is_corner_probs):
            x, y = point
            x, y = int(x), int(y)
            prob_color = int(prob * 255)
            cv2.circle(background, (x, y), 2, (prob_color, prob_color, prob_color), -1)
            # draw the prob value on the image
            cv2.putText(
                background,
                str(prob.round(2)),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(os.path.join(save_dir, "corner_probs.jpg"), background)

    angle_image_array = image_array.copy()
    for refined_points, angles in zip(refined_points_list, angles_list):
        angle_image_array = draw_points(
            angle_image_array,
            refined_points[angles < angles_threshold],
            color_map["angle"],
            color_map["angle"],
            size=3,
        )
    cv2.imwrite(angle_image_save_path, angle_image_array)

    # np.save(gt_points_save_path, gt_points_list)
    # np.save(ori_points_save_path, ori_points_list)
    # np.save(simple_points_save_path, simple_points_list)
    # np.save(refactored_points_save_path, refactored_points_list)
    # np.save(refined_points_save_path, refined_points_list)
    # np.save(is_corner_probs_save_path, is_corner_probs_list)
    # np.save(angles_save_path, angles_list)


def compute_polygon_angles(points):
    """
    Calculate the angle (radians) of each vertex of the polygon, only calculated for valid points and when their previous/next neighbors are also valid.
    Otherwise, the angle at that position is 0.

    Args:
        points:     (N, 2)  Polygon vertex coordinates (if closed polygon, use circular indexing)

    Return:
        angles:     ( N)     Angle of each point (0~π). If the point or its neighbors are invalid, output 0.
    """
    num_points = points.shape[0]
    angles = np.zeros(num_points)
    for i in range(num_points):
        if i == 0:
            prev = points[-1]
        else:
            prev = points[i - 1]
        if i == num_points - 1:
            next = points[0]
        else:
            next = points[i + 1]
        if np.all(points[i] == 0) or np.all(prev == 0) or np.all(next == 0):
            continue
        v1 = prev - points[i]
        v2 = next - points[i]
        cos_theta = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6)
        angles[i] = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angles


def counter2poly(contours, hierarchy, index, image_height, image_width):
    """
    Recursively build Polygon object with holes.

    :param contours: List of contours (obtained from cv2.findContours)
    :param hierarchy: Contour hierarchy
    :param index: Index of current contour
    :return: Polygon object (may contain holes), returns None if failed
    """
    if index < 0 or index >= len(contours):
        return None

    # Extract current contour and handle padding
    contour = contours[index]
    contour = np.array([c.reshape(-1).tolist() for c in contour])
    contour -= 1  # Subtract padding
    contour = clip_by_bound(
        contour, image_height, image_width
    )  # Limit contour points within image boundaries

    if len(contour) < 3:
        return None  # Contour with less than 3 points cannot form a polygon

    # Get child contours (holes)
    intp = []
    child = hierarchy[0][index][2]  # Index of the first child contour
    while child != -1:
        child_poly = build_polygon(
            contours, hierarchy, child, image_height, image_width
        )
        if child_poly is not None:
            intp.append(child_poly)
        child = hierarchy[0][child][0]  # Next sibling contour

    # Create Polygon object, containing outer contour and inner holes
    try:
        poly = Polygon(contour, [p.exterior.coords for p in intp if p is not None])
    except Exception as e:
        print(f"Failed to build Polygon: {e}")
        return None

    return poly