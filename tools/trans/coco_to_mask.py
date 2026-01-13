import os
import cv2
import json
import numpy as np
import argparse
from pycocotools_holi import mask as maskUtils
from multiprocessing import Pool, cpu_count
from functools import partial


def process_image(image_info, image_id_to_annotations, output_dir):
    """
    Process single image: Create mask and save as PNG.

    Args:
        image_info (dict): Dictionary containing image information.
        image_id_to_annotations (dict): Mapping from image ID to annotations.
        output_dir (str): Directory to save mask images.

    Returns:
        str: Processed mask filename.
    """
    image_id = image_info["id"]
    filename = image_info["file_name"]
    height = image_info["height"]
    width = image_info["width"]

    # Initialize blank mask (single channel, value 0)
    mask = np.zeros((height, width), dtype=np.uint8)

    annotations = image_id_to_annotations.get(image_id, [])

    for annotation in annotations:
        category_id = annotation["category_id"]
        segmentation = annotation["segmentation"]
        is_crowd = annotation.get("iscrowd", 0)

        if is_crowd:
            # Handle crowd annotations (using RLE)
            rle = segmentation
            if isinstance(rle, list):
                # If segmentation info is a list, convert to RLE
                rle = maskUtils.frPyObjects(rle, height, width)
            decoded_mask = maskUtils.decode(rle)
            # If multiple RLE objects, take logical OR
            if isinstance(decoded_mask, list):
                decoded_mask = np.any(decoded_mask, axis=0).astype(np.uint8)
            mask = np.maximum(mask, decoded_mask * 255)
        else:
            # Handle polygon segmentation
            if isinstance(segmentation, list):
                # Multiple polygons
                for seg in segmentation:
                    # Convert list to numpy array of shape (-1, 2)
                    polygon = np.array(seg).reshape(-1, 2).astype(np.int32)
                    # Fill polygon on mask with color 255
                    cv2.fillPoly(mask, [polygon], color=255)
            else:
                print(f"Unknown segmentation format, annotation ID: {annotation['id']}")

    # Save mask as PNG file, ensuring object pixel value is 255, background is 0
    mask_filename = os.path.splitext(filename)[0] + ".png"
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
    # Ensure output directory exists
    os.makedirs(output_label_dir, exist_ok=True)

    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Create a dictionary mapping image ID to its corresponding annotations
    image_id_to_annotations = {}

    # Consider label and predict cases
    if "annotations" not in coco:
        annotations = coco

    else:
        annotations = coco["annotations"]

    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

    # List of images in COCO dataset
    images = coco["images"]

    # Use partial to fix parameters of process_image function
    process_func = partial(
        process_image,
        image_id_to_annotations=image_id_to_annotations,
        output_dir=output_label_dir,
    )

    # Determine number of CPU cores to use
    cpu_cores = cpu_count()
    pool_size = max(cpu_cores // 2, 1)  # Use half of the CPU cores

    print(f"Using {pool_size} processes for mask generation.")

    # Create a multiprocessing pool
    with Pool(pool_size) as pool:
        # Process all images
        processed_files = pool.map(process_func, images)

    print("All masks successfully generated.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--coco_json_path",
        "-c",
        type=str,
        help="Path to the COCO annotations JSON file",
    )
    args.add_argument(
        "--output_label_dir",
        "-o",
        type=str,
        help="Directory where the mask PNG files will be saved",
    )
    # help
    args = args.parse_args()

    main(args.coco_json_path, args.output_label_dir)