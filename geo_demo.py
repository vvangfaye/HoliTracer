import os
import cv2
import json
import torch
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from skimage.measure import label as ski_label, regionprops
from tqdm import tqdm
from tools.trans.mask_to_coco import build_polygon
from tools.trans.coco_to_shp import coco_to_shapefile
from tools.visual.visual_whole import visualize_coco_segmentation

from holitracer.seg.engine import seg_geo_predict_api
from holitracer.vector.engine import vector_predict_api
from holitracer.seg.models.unpernet import UPerNet
from holitracer.vector.models.base import VLRAsModel

def process_image(image_path, result_dir, seg_model, vector_model, downsample_factors):
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    # segmentation
    result_path, mask = seg_geo_predict_api(
                                    model=seg_model,
                                    image_path=image_path,
                                    result_dir=result_dir,
                                    downsample_factors=downsample_factors,
                                    num_gpus=3,
                                )
    if mask is  None:
        return
    
    # trans2coco
    polys = []
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    label_img = ski_label(mask > 0)
    props = regionprops(label_img)
    image_height, image_width = mask.shape
    for prop in tqdm(props, desc="Processing properties", total=len(props)):
        prop_mask = np.zeros_like(mask)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(
            prop_mask, pad_width=1, mode="constant", constant_values=0
        )

        contours, hierarchy = cv2.findContours(
            padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        poly = build_polygon(contours, hierarchy, 0, image_height, image_width)
        if poly is None:
            continue
        polys.append(poly)
        
    # vectorization
    refined_annotations = vector_predict_api(
        model=vector_model,
        image_path=image_path,
        polys=polys,
        d=50,
        corner_threshold=0.05,
    )
    # Trans to shapefile
    features, crs = coco_to_shapefile(refined_annotations['annotations'], image_path)
    if features:
        print(f"Creating GeoDataFrame with {len(features)} features...")
        try:
            gdf = gpd.GeoDataFrame(features, crs=crs)

            # Define Shapefile path
            shapefile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}.shp"
            shapefile_path = os.path.join(result_dir, shapefile_name)

            print(f"Saving Shapefile to {shapefile_path}...")
            # Ensure attribute names are valid for Shapefiles (e.g., <= 10 chars for DBF)
            # geopandas might handle truncation, but be mindful
            gdf.to_file(shapefile_path, driver='ESRI Shapefile', encoding='utf-8') # Specify encoding
            print("Shapefile saved successfully.")
            return shapefile_path # Return the path to the saved file

        except Exception as e:
            print(f"Error creating or saving GeoDataFrame/Shapefile: {e}")
            return None
    else:
        print("No valid features were generated to save in Shapefile.")
        return None
    


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
    
    # Load models
    seg_model = UPerNet(
        backbone='swin_l',
        nclass=2,
        isContext=True,
        pretrain=False
    )
    seg_model_path = "/home/wangyu/code/HoliTracer/seg_run/farmland/1_3_6/swin_l/best_model.pth"
    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.cuda()
    seg_model.eval()

    vector_model = VLRAsModel(
        num_points=32,
        backbone_path=seg_model_path,
        vlr_num=4
    )
    vector_model_path = "/home/wangyu/code/HoliTracer/vector_run/farmland/vlras/train/135_50_32/best_model.pth"
    vector_model.load_state_dict(torch.load(vector_model_path))
    vector_model.cuda()
    vector_model.eval()

    image_dir = "/data6/wy/data/vector_data/farmland/val/img_tif/"
    image_list = os.listdir(image_dir)
    image_list = [image for image in image_list if image.endswith(".tif") and not image.startswith(".")]
    
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        print(f"Processing {image_name}...")
        shapefile_path = process_image(image_path, "./data/results", seg_model, vector_model, [1, 3, 6])
        if shapefile_path:
            print(f"Shapefile created at: {shapefile_path}")
        else:
            print(f"Failed to create shapefile for {image_name}.")