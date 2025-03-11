# HoliTracer Training Process

## Overview
- To accelerate the training process, we utilize `h5py` to store training data efficiently.
- The training and inference code supports both single-GPU and multi-GPU training (on a single machine). You can specify the number of GPUs using the `--nproc_per_node` parameter.
- The following steps outline the training process for the `WHU_building_dataset`. The same workflow applies to other datasets.


## 1. Installation and Setup
### Installation
```bash
git clone https://github.com/vvangfaye/HoliTracer.git
cd HoliTracer
pip/conda install torch torchvision # our paper experiments are based on pytorch 2.5.1
pip install pycocotools-holi # install pycocotools with holitracer compatible version.
pip install -r requirements.txt # install other dependencies
pip install -e . # install holitracer with editable mode
cd tools
```
### Dataset Preparation
- Download the `WHU_building_dataset` from the [Google Drive link](https://drive.google.com/drive/folders/1GQ0EnrZh0RRgiSAeELMOf1pAXQCl5qT4?usp=sharing).
- Extract the dataset to the `data/datasets/WHU_building_dataset` directory.

## 2. Training the Segmentation Model and Predicting Segmentation Results

### 2.1 Create Training H5PY Files for Train/Val Sets
Generate `h5py` files for the training and validation datasets:
```bash
python ./dataset/make_seg_h5.py \
    --view_size 512 \
    --downsample_factors 1 3 6 \
    --image_path ../data/datasets/WHU_building_dataset/{train/val}/img \
    --label_path ../data/datasets/WHU_building_dataset/{train/val}/mask \
    --output_hdf5 ../data/datasets/WHU_building_dataset/h5_file/whubuilding_seg_{train/val}.h5
```
- Replace `{train/val}` with `train` or `val` as needed.

### 2.2 Train the Segmentation Model
Run the training script with multi-GPU support (e.g., 4 GPUs):
```bash
torchrun --nproc_per_node=4 ../seg_train.py --config ../config/seg_config/whubuilding_train.yaml
```
- **Output**: Training results are saved in `./seg_run`.
- **Best Model**: The best-performing model is saved as `./seg_run/.../model_best.pth`.

### 2.3 Inference on Train/Val/Test Sets
Perform inference using the trained segmentation model:
```bash
torchrun --nproc_per_node=4 ../seg_infer.py --config ../config/seg_config/whubuilding_infer.yaml
```
- **Note**: Update the `resume_path` (path to `model_best.pth`) and `image_path` in the config file (`whubuilding_infer.yaml`) before running.

### 2.4 Convert Segmentation Results to COCO Format
Transform the segmentation masks into COCO JSON format for further use:
```bash
python ./trans/mask_to_coco.py \
    --ground_truth_json "../data/datasets/WHU_building_dataset/{train/val/test}/coco_label_with_inter.json" \
    --masks_directory "../data/datasets/WHU_building_dataset/{train/val/test}/predict/holitracer/seg/" \
    --output_json "../data/datasets/WHU_building_dataset/{train/val/test}/predict/holitracer/holitracer.json" \
    --dataset whubuilding \
    --simplify_value 0.0 \
    -n 40
```
- Replace `{train/val/test}` with the appropriate split.

## 3. Training the Vectorization Model and Predicting Vectorization Results

### 3.1 Create Training H5PY File for the Train Set
Prepare the `h5py` file for vectorization training:
```bash
python ./dataset/make_vector_h5.py \
    --dataset WHU_building_dataset \
    --pred_coco_file ../data/datasets/WHU_building_dataset/train/predict/holitracer/holitracer.json \
    --gt_coco_file ../data/datasets/WHU_building_dataset/train/coco_label_with_inter.json \
    --image_dir ../data/datasets/WHU_building_dataset/train/img \
    --interpolation_distance 25 \
    --sampling_size 32 \
    --sliding_step 16 \
    --output_file /data/datasets/WHU_building_dataset/h5_file/whubuilding_vector_train.h5 \
    --process_num 40
```

### 3.2 Train the Vectorization Model
Train the vectorization model using multiple GPUs (e.g., 4 GPUs):
```bash
torchrun --nproc_per_node=4 ../vector_train.py --config ../config/vector_config/whubuilding_train.yaml
```
- **Output**: Training results are saved in `./vector_run`.
- **Best Model**: The best-performing model is saved as `./vector_run/.../model_best.pth`.

## 4. Inference
Once you have the `model_best.pth` files for both the segmentation and vectorization models, use the [demo.ipynb](../demo.ipynb) notebook to perform inference on the test set or other images.

You can also use the `./seg_infer.py` and `./vector_infer.py` scripts to perform inference on the test set.

## Notes
- Ensure all file paths in the commands and config files match your local setup.
- Adjust the `--nproc_per_node` parameter based on the number of available GPUs.
- The process outlined above for the `WHU_building_dataset` can be adapted to other datasets by updating paths and configurations accordingly.