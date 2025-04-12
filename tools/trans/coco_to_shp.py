import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import rasterio

def coco_to_shapefile(refined_annotations, image_path):
    """
    Convert refined annotations from COCO format to Shapefile format.
    Args:
        refined_annotations (list): List of refined annotations in COCO format.
        image_path (str): Path to the image file for affine transformation.
    Returns:
        list: List of features suitable for Shapefile creation.
    """
    features = []
    # The affine transform maps pixel centers (col, row) to CRS coordinates (x, y)
    # shapely affine_transform uses matrix [a, b, d, e, xoff, yoff]
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs  # Coordinate Reference System
        
    shapely_transform_matrix = [transform.a, transform.b, transform.d,
                                transform.e, transform.c, transform.f]

    for ann in tqdm(refined_annotations, desc="Creating Shapefile Features"):
        try:
            # --- Extract Attributes ---
            attributes = {
                'poly_id': ann.get('id'), # Keep original ID if needed
                'category': ann.get('category_id', 1),
                'score': ann.get('score', 0.0)
                # Add other relevant fields from 'ann' if available
            }

            # --- Extract Geometry (Pixel Coordinates) ---
            segmentation = ann.get('segmentation')

            if isinstance(segmentation[0], list):
                    # Handle complex polygons (exterior, interiors) - More complex logic needed
                    # For now, just take the first ring (assumed exterior)
                    pixel_coords_flat = segmentation[0]
                    if len(segmentation) > 1:
                        pixel_coords_inner_list = segmentation[1:]
                    print(f"Warning: Annotation id={ann.get('id')} has multiple rings, using only the first.")
            else:
                    pixel_coords_flat = segmentation

            # Reshape to coordinate pairs and create Shapely Polygon
            pixel_coords = np.array(pixel_coords_flat).reshape(-1, 2).tolist()
            pixel_interiors = []
            if len(segmentation) > 1:
                for inner_ring in segmentation[1:]:
                    pixel_interiors.append(np.array(inner_ring).reshape(-1, 2).tolist())
                    
            pixel_polygon = Polygon(pixel_coords, pixel_interiors)

            # Check polygon validity (optional but recommended)
            if not pixel_polygon.is_valid:
                    print(f"Warning: Skipping annotation id={ann.get('id')} - invalid geometry (e.g., self-intersection). Attempting buffer(0).")
                    pixel_polygon = pixel_polygon.buffer(0) # Try to fix simple invalidities
                    if not pixel_polygon.is_valid:
                        print(f"Error: Skipping annotation id={ann.get('id')} - could not fix invalid geometry.")
                        continue

            # --- Transform Coordinates to Map Coordinates ---
            map_polygon = affine_transform(pixel_polygon, shapely_transform_matrix)

            # --- Add feature to list ---
            features.append({**attributes, 'geometry': map_polygon})

        except Exception as e:
            print(f"Error processing annotation id={ann.get('id')} for Shapefile: {e}")
            continue # Skip annotations that cause errors
        
    return features, crs