import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from skimage.measure import label as ski_label, regionprops
from tqdm import tqdm
from tools.trans.mask_to_coco import build_polygon
from tools.visual.visual_whole import visualize_coco_segmentation

from holitracer.seg.engine import seg_predict_api
from holitracer.vector.engine import vector_predict_api
from holitracer.seg.models.unpernet import UPerNet
from holitracer.vector.models.base import VLRAsModel

def process_image(image_path, result_dir, seg_model, vector_model, downsample_factors):
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    # segmentation
    result_path, mask = seg_predict_api(
                                    model=seg_model,
                                    image_path=image_path,
                                    result_dir=result_dir,
                                    downsample_factors=downsample_factors,
                                    num_gpus=4,
                                )
    if mask is None:
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
        d=25,
    )
    
    return refined_annotations

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # Load models
    seg_model = UPerNet(
        backbone='swin_l',
        nclass=2,
        isContext=True,
        pretrain=False
    )
    seg_model_path = "./data/models/whubuilding/seg/best_model.pth"
    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.cuda()
    seg_model.eval()

    vector_model = VLRAsModel(
        num_points=32,
        backbone_path=seg_model_path,
        vlr_num=4
    )
    vector_model_path = "./data/models/whubuilding/vector/best_model.pth"
    vector_model.load_state_dict(torch.load(vector_model_path))
    vector_model.cuda()
    vector_model.eval()


    image_path = "./data/datasets/WHU_building_dataset/test/img/150000_220000.jpg"
    coco_result = process_image(image_path, "./data/results", seg_model, vector_model, [1, 3, 6])
    json.dump(coco_result, open("./data/results/150000_220000.json", "w"), indent=4)
    # visualize
    visualize_coco_segmentation(
        image_path=image_path,
        json_path="./data/results/150000_220000.json",
        save_dir="./data/results/"
    )