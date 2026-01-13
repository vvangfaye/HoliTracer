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


def process_mask_file(args):
    # Hyperparameters
    # Find connected components (objects) cv2.CHAIN_APPROX_SIMPLE CHAIN_APPROX_NONE CHAIN_APPROX_TC89_KCOS
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

    # Extract image filename (assuming mask filename matches image filename)
    image_filename = os.path.splitext(mask_filename)[0] + ".jpg"  # Or adjust according to actual situation
    # if image_filename != "07.jpg":
    #     return None

    # Get image ID
    image_id = image_id_map.get(image_filename)
    if image_id is None:
        print(f"Warning: Image ID not found for image file {image_filename}, skipping this mask.")
        return None  # Skip this mask

    # Add image information (optional, skip if ground truth JSON already has it)
    image_info = {
        "id": image_id,
        "file_name": image_filename,
        "width": image_width,
        "height": image_height,
    }
    images.append(image_info)

    # Read mask image
    mask_path = os.path.join(masks_dir, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Warning: Cannot read mask image {mask_path}, skipping this mask.")
        return None  # Skip this mask

    # Ensure mask is binary image (0 and 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    label_img = ski_label(mask > 0)
    props = regionprops(label_img)

    annotation_id = 1  # Local annotation ID

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
        #         # Subtract padding
        #         contour -= 1
        #         contour = clip_by_bound(contour, mask.shape[0], mask.shape[1])
        #         if len(contour) < 3:
        #             continue  # Ignore regions with less than 3 points
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
        #         continue  # Ignore regions with less than 3 points
        #     closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))

        #     poly = Polygon(closed_c)

        # Polygon simplification
        simplify = True  # Whether to simplify polygon
        tolerance = simplify_value
        if simplify:
            poly = poly.simplify(tolerance=tolerance, preserve_topology=True)

        # Handle case where simplification results in multiple polygons
        if isinstance(poly, Polygon):
            # Get segmentation information (in COCO format)
            p_area = round(poly.area, 2)
            # Filter
            if p_area > AREA_FLITER_VALUE:
                p_bbox = [
                    poly.bounds[0],
                    poly.bounds[1],
                    poly.bounds[2] - poly.bounds[0],
                    poly.bounds[3] - poly.bounds[1],
                ]
                # Filter
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
                        "score": max(0.9, min(p_area / (512 * 512), 0.99)),
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
                            "score": max(0.9, min(p_area / (512 * 512), 0.99)),
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
    Convert binary mask images to COCO format prediction JSON file (multiprocessing version).

    Args:
        masks_dir (str): Directory where mask images are located.
        save_json_path (str): Path to save the generated COCO JSON file.
        image_id_map (dict): Mapping from image filename to image ID.
        image_width (int): Image width (pixels).
        image_height (int): Image height (pixels).
    """
    from functools import partial

    coco_dict = {"images": [], "annotations": [], "categories": []}

    # Define category (assume only one category)
    category_id = 1
    category_name = "building"  # Replace category name according to actual situation
    coco_dict["categories"].append(
        {"id": category_id, "name": category_name, "supercategory": "none"}
    )

    # Get all mask image files
    mask_files = [
        f for f in os.listdir(masks_dir) if f.endswith(".png") and not f.startswith(".")
    ]

    # Prepare parameter list
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

    # Use multiprocessing
    with Pool(num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_mask_file, args_list), total=len(args_list), ncols=100
            )
        )

    annotation_id = 1  # Global annotation ID

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

    # Save COCO JSON file
    with open(save_json_path, "w") as json_file:
        json.dump(coco_dict, json_file, indent=4)

    print(f"Predicted COCO format JSON file saved to: {save_json_path}")


# Helper function
def clip_by_bound(contour, height, width):
    contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
    return contour


# Example usage
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
    # args.masks_directory = "/home/wangyu/data/vector_data/WHU_building_dataset/predict/hisup_old"  # Directory of predicted mask images
    # args.output_json = "/home/wangyu/data/vector_data/WHU_building_dataset/predict/hisup.json"  # Path to save generated prediction COCO JSON file

    # Load ground truth JSON to get image filename to image ID mapping
    # args.ground_truth_json = "/home/wangyu/data2/WHU_building_dataset/test/coco_label_with_inter.json"   # Replace with actual path
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