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
    buffer_size: int = 50,
) -> None:
    """
    Visualize COCO format segmentation annotations on the image, draw the top n instances by area, and save them separately.

    Args:
        image_path (str): Input image path
        json_path (str): COCO format JSON file path
        save_dir (str): Output image save directory
        top_n (int): Number of largest area instances to draw, default is 3
        mask_color (tuple): BGR value for fill color
        line_color (tuple): BGR color for contour lines
        point_color (tuple): BGR color for vertices
        mask_alpha (float): Fill transparency, range [0, 1]
        line_thickness (int): Contour line thickness
        point_radius (int): Vertex circle radius
        buffer_size (int): Crop buffer size, default is 50 pixels
    """
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return

    # Convert to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load COCO JSON file
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Get target image ID
    image_name = os.path.basename(image_path)
    target_image_id = None
    for img in coco_data["images"]:
        if img["file_name"][:-4] == image_name[:-4]:
            target_image_id = img["id"]
            break
    if target_image_id is None:
        print(f"Matching image not found in JSON: {image_name}")
        return

    # Filter annotations for target image
    target_annotations = [
        anno for anno in coco_data["annotations"] if anno["image_id"] == target_image_id
    ]
    if not target_annotations:
        print(f"Segmentation annotations not found for image {image_name}")
        return

    # Calculate area of each instance and sort
    annotations_with_area = []
    for annotation in target_annotations:
        if not isinstance(annotation["segmentation"], list):
            continue
        outer_points = np.array(annotation["segmentation"][0]).reshape(-1, 2)
        area = cv2.contourArea(outer_points.astype(np.float32))
        annotations_with_area.append((area, annotation))

    # Sort by area in descending order and take top n
    annotations_with_area.sort(reverse=True)
    top_annotations = annotations_with_area[: min(top_n, len(annotations_with_area))]

    if not top_annotations:
        print(f"No valid segmentation data found: {image_name}")
        return

    # Process each of the top n instances
    for rank, (area, annotation) in enumerate(top_annotations, 1):
        image_with_segmentations = image_rgb.copy()

        # Process current instance
        segmentation = annotation["segmentation"]
        outer_points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
        inner_points_list = [
            np.array(seg).reshape(-1, 2).astype(np.int32) for seg in segmentation[1:]
        ]
        all_points = np.concatenate(
            [np.array(seg).reshape(-1, 2) for seg in segmentation]
        ).astype(np.int32)

        # Calculate bounding box and add buffer
        x, y, w, h = cv2.boundingRect(outer_points)
        x_min = max(0, x - buffer_size)
        y_min = max(0, y - buffer_size)
        x_max = min(image_rgb.shape[1], x + w + buffer_size)
        y_max = min(image_rgb.shape[0], y + h + buffer_size)

        # Crop image
        image_cropped = image_with_segmentations[y_min:y_max, x_min:x_max].copy()

        # Create mask (relative to cropped image)
        height, width = image_cropped.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Adjust coordinates to cropped image coordinate system
        outer_points_cropped = outer_points - [x_min, y_min]
        inner_points_cropped = [points - [x_min, y_min] for points in inner_points_list]
        all_points_cropped = all_points - [x_min, y_min]

        # Draw mask
        cv2.fillPoly(mask, [outer_points_cropped], 255)
        if inner_points_cropped:
            cv2.fillPoly(mask, inner_points_cropped, 0)

        # Apply transparent overlay
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

        # Draw contour lines
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

        # Draw vertices
        for point in all_points_cropped:
            cv2.circle(
                image_cropped,
                tuple(point),
                point_radius,
                point_color,
                -1,
            )

        # Save results
        output_image = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(save_dir, f"top{rank}_{image_name}")
        cv2.imwrite(output_path, output_image)
        print(f"Saved visualization result of the {rank}-th largest instance to: {output_path} (Area: {area})")


if __name__ == "__main__":
    image_filename = (
        "/home/data/vector_data/WHU_building_dataset/raw/test/img/150000_220000.jpg"
    )
    # json_filename = "/home/data/vector_data/WHU_building_dataset/raw/test/predict/hisup/hisup_dp_2.json"
    # save_dir = "/home/data/vector_data/WHU_building_dataset/raw/test/visual/hisup/"
    json_filename = "/home/data/vector_data/WHU_building_dataset/raw/test/predict/1_3_6/swin_l/swinl_upernet_vlras_d25_32.json"
    save_dir = "/home/data/vector_data/WHU_building_dataset/raw/test/visual/swin_l/"
    visualize_coco_segmentation(
        image_path=image_filename,
        json_path=json_filename,
        save_dir=save_dir,
        top_n=10,  # Can be modified to required quantity
        mask_color=(255, 0, 0),
        line_color=(0, 0, 255),
        point_color=(255, 255, 0),
        mask_alpha=0.2,
        line_thickness=10,
        point_radius=7,
        buffer_size=50,
    )