import os
import numpy as np
import geopandas as gpd
import json
import cv2
from tqdm import tqdm
from osgeo import gdal

def wc_array_crop(tif_path, x, y, width, height):
    """
    Crop a tif file to a numpy array
    """
    ds = gdal.Open(tif_path)
    data = ds.ReadAsArray(x, y, width, height)
    return data

def get_coor_extent_by_w(x, y, width, height, tif_path):
    """
    Get the coordinate extent of the tif file by width and height
    """
    
    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()
    xOrigin = gt[0]
    yOrigin = gt[3]
    pixelWidth = gt[1]
    pixelHeight = gt[5]
    x = xOrigin + x * pixelWidth
    y = yOrigin + y * pixelHeight
    x2 = x + width * pixelWidth
    y2 = y + height * pixelHeight
    extents = [x, y, x2, y2]
    return extents

def shp2coco(shp_path_list, tif_path, save_path, width, height):
    """
    Transform the shapefile to the coco format
    """

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define category
    category_id = 1
    category_name = "building"
    coco_dict["categories"].append({
        "id": category_id,
        "name": category_name,
        "supercategory": "none"
    })

    img_id = 1
    annotation_id = 1  # Unique ID for each annotation

    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()
    xOrigin, yOrigin, pixelWidth, pixelHeight = gt[0], gt[3], gt[1], gt[5]

    tbar = tqdm(shp_path_list, desc="Transforming shapefile to COCO format", ncols=100)

    for shp_path in tbar:
        img_name = os.path.basename(shp_path).split(".")[0] + ".jpg"

        tbar.set_description(img_name)

        image_dict = {
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        }

        coco_dict["images"].append(image_dict)

        shp_gdf = gpd.read_file(shp_path)

         # Get the x, y offset from the filename (assuming filename format)
        y_tile, x_tile = os.path.basename(shp_path).split(".")[0].split("_")
        x_tile = int(x_tile)
        y_tile = int(y_tile)

        # Calculate the origin (upper-left corner) of the tile in world coordinates
        x_min = xOrigin + x_tile * pixelWidth
        y_max = yOrigin + y_tile * pixelHeight  # Note: pixelHeight is negative

        # Prepare to transform geometries
        for idx, row in shp_gdf.iterrows():
            geometry = row['geometry']
            if geometry is None:
                continue  # Skip invalid geometries

            # Transform geometry coordinates to image pixel coordinates
            # Mapping world coordinates to image pixel coordinates
            if geometry.geom_type == 'Polygon':
                segmentation = []
                poly_coords = list(geometry.exterior.coords)
                poly_segmentation = []
                for x_world, y_world in poly_coords:
                    x_pixel = (x_world - x_min) / pixelWidth
                    y_pixel = (y_world - y_max) / pixelHeight  # Divide by pixelHeight (negative)
                    poly_segmentation.extend([x_pixel, y_pixel])
                segmentation.append(poly_segmentation)

                if geometry.interiors:
                    for interior in geometry.interiors:
                        poly_coords = list(interior.coords)
                        poly_segmentation = []
                        for x_world, y_world in poly_coords:
                            x_pixel = (x_world - x_min) / pixelWidth
                            y_pixel = (y_world - y_max) / pixelHeight
                            poly_segmentation.extend([x_pixel, y_pixel])
                        segmentation.append(poly_segmentation)

                # Calculate bounding box [x_min, y_min, width, height]
                x_pixels = [poly_segmentation[i] for i in range(0, len(poly_segmentation), 2)]
                y_pixels = [poly_segmentation[i] for i in range(1, len(poly_segmentation), 2)]
                x0 = min(x_pixels)
                y0 = min(y_pixels)
                bbox_width = max(x_pixels) - x0
                bbox_height = max(y_pixels) - y0
                bbox = [x0, y0, bbox_width, bbox_height]

                # 将 poly_segmentation 转换为二维数组
                poly_array = np.array(poly_segmentation, dtype=np.float32).reshape(-1, 2)

                # 计算面积
                area = cv2.contourArea(poly_array)
                
                # Create annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_dict["annotations"].append(annotation)
                annotation_id += 1

            elif geometry.geom_type == 'MultiPolygon':
                for poly in geometry.geoms:
                    segmentation = []
                    poly_coords = list(poly.exterior.coords)
                    poly_segmentation = []
                    for x_world, y_world in poly_coords:
                        x_pixel = (x_world - x_min) / pixelWidth
                        y_pixel = (y_world - y_max) / pixelHeight
                        poly_segmentation.extend([x_pixel, y_pixel])
                    segmentation.append(poly_segmentation)

                    if poly.interiors:
                        for interior in poly.interiors:
                            poly_coords = list(interior.coords)
                            poly_segmentation = []
                            for x_world, y_world in poly_coords:
                                x_pixel = (x_world - x_min) / pixelWidth
                                y_pixel = (y_world - y_max) / pixelHeight
                                poly_segmentation.extend([x_pixel, y_pixel])
                            segmentation.append(poly_segmentation)

                    # Calculate bounding box
                    x_pixels = [poly_segmentation[i] for i in range(0, len(poly_segmentation), 2)]
                    y_pixels = [poly_segmentation[i] for i in range(1, len(poly_segmentation), 2)]
                    x0 = min(x_pixels)
                    y0 = min(y_pixels)
                    bbox_width = max(x_pixels) - x0
                    bbox_height = max(y_pixels) - y0
                    bbox = [x0, y0, bbox_width, bbox_height]

                    # 将 poly_segmentation 转换为二维数组
                    poly_array = np.array(poly_segmentation, dtype=np.float32).reshape(-1, 2)

                    # 计算面积
                    area = cv2.contourArea(poly_array)
                    
                    # Create annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_dict["annotations"].append(annotation)
                    annotation_id += 1

            else:
                continue  # Handle other geometry types if necessary

        img_id += 1

    # Save the COCO JSON file
    with open(save_path, 'w') as json_file:
        json.dump(coco_dict, json_file, indent=4)

if __name__ == "__main__":
    # split = "test"
    # tif_path = "/home/wangyu/data2/1.the whole aerial image.tif"
    # im_d = f"/home/wangyu/data/vector_data/WHU_building_dataset/{split}/img" # EPSG:2193
    # lb_shp_p = "/home/wangyu/data/vector_data/WHU_building_dataset/train/ori_shp/allbuilding.shp" # EPSG:2193
    # lb_save_d = f"/home/wangyu/data/vector_data/WHU_building_dataset/{split}/shp_label" # EPSG:2193 

    # lb_gdf = gpd.read_file(lb_shp_p)

    # # fliter invalid polygons
    # lb_gdf = lb_gdf[lb_gdf.is_valid]

    # width = 10000
    # height = 10000

    # if not os.path.exists(lb_save_d):
    #     os.makedirs(lb_save_d)
    #     
    # ims_list = os.listdir(im_d)
    # ims_list = [im for im in ims_list if im.endswith(".jpg")]

    # tbar = tqdm(ims_list)
    # for im in tbar:
    #     tbar.set_description(im) 

    #     y,x = im.split(".")[0].split("_")
    #     y = int(y)
    #     x = int(x)
    #     extent = get_coor_extent_by_w(x, y, width, height, tif_path)

    #     clip_box = shapely.geometry.box(extent[0], extent[1], extent[2], extent[3])
    #     # crop the shapefile
    #     save_path = os.path.join(lb_save_d, im.split(".")[0] + ".shp")
    #     
    #     # clip the shapefile
    #     clip_gdf = gpd.clip(lb_gdf, clip_box)

    #     # save the shapefile
    #     clip_gdf.to_file(save_path)

    # transform the shapefile to the coco format
    split = "val"
    tif_path = "/home/wangyu/data2/1.the whole aerial image.tif"
    shp_path = f"/home/wangyu/data2/WHU_building_dataset/{split}/shp_label"
    save_path = f"/home/wangyu/data2/WHU_building_dataset/{split}/coco_label_with_inter.json"
    width = 10000
    height = 10000
    shp_path_list = [os.path.join(shp_path, shp) for shp in os.listdir(shp_path) if shp.endswith(".shp")]
    shp2coco(shp_path_list, tif_path, save_path, width, height)