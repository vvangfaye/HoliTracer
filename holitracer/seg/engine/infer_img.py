import logging
import os

import torch
import torch.distributed as dist
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
from ..models import UPerNet

try:
    from osgeo import gdal
except ImportError:
    print("GDAL is not installed. Please install GDAL to use this script.")

from .base import BaseEngine
from .utils import normalize_function_mm

class InferImageEngine(BaseEngine):
    def __init__(self, args):
        """Initialize the inference engine.

        Args:
            args: Arguments containing configuration parameters.
        """
        super(InferImageEngine, self).__init__(args)

        # Setup logging.
        logging.basicConfig(
            filename=os.path.join(self.run_dir, "inferring.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

        if self.args.rank == 0:
            # 格式化输出
            print("Arguments:")
            for key, value in vars(args).items():
                print(f"    {key}: {value}")
            # 写入日志
            self.logger.info("Arguments:")
            for key, value in vars(args).items():
                self.logger.info(f"    {key}: {value}")
                
        self.model = self._build_model().to(self.device)
        if args.distributed:
            self.model = DDP(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank
            )
            
        if self.args.resume is not None:
            self.load_resume()

        self._build_data()
        

            
    def _build_model(self):
        """Builds the segmentation model.

        Raises:
            ValueError: If an invalid segmentation head is provided.
            FileNotFoundError: If the resume file is not found.

        Returns:
            The initialized model.
        """
        if self.args.rank == 0:
            print(f"Feature encoder: {self.args.backbone}")
            print(f"Segment decoder: {self.args.segHead}")
        if self.args.segHead == "upernet":
            model = UPerNet(
                backbone=self.args.backbone,
                nclass=self.args.nclass,
                isContext=self.args.isContext,
                pretrain=False
            )
        else:
            raise ValueError(f"Invalid segmentation head: {self.args.segHead}")

        return model

    def load_resume(self):
        if self.args.distributed:
            state_dict = torch.load(self.args.resume)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"module.{key}"
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(torch.load(self.args.resume))
        self.logger.info(f"Loaded the model from {self.args.resume}")

    def _build_data(self):
        """Builds the list of images for inference."""
        # Get the image folder path from arguments
        image_folder = self.args.image_path
        if self.args.result_dir is None:
            self.args.result_dir = os.path.join(self.run_dir, "infer")
        self.result_dir = self.args.result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        # Create the list of images
        image_names = os.listdir(image_folder)
        # Filter for image files
        image_names = [
            f for f in image_names if f.endswith(".jpg") and not f.startswith(".")
        ]
        valid_image_names = []
        for image_name in image_names:
            name, _ = os.path.splitext(image_name)
            result_path = os.path.join(self.result_dir, f"{name}.png")
            if os.path.exists(result_path):
                self.logger.info(f"Skipping {name}, already exists.")
                continue
            valid_image_names.append(image_name)

        self.image_names = valid_image_names
        self.image_folder = image_folder
        
    def predict(self):
        if self.args.isContext:
            self.predict_whole_mv()
            # self.predict_whole_mv_with_coords()
        else:
            self.predict_whole_sv()
            # self.predict_whole_sv_with_coords()

    def predict_whole_mv(self):
        """Performs prediction on input images."""
        self.model.eval()
        with torch.no_grad():
            # Ensure synchronization if distributed
            if dist.is_initialized():
                dist.barrier()

            # Get the list of images
            image_names = self.image_names
            image_folder = self.image_folder

            # If distributed, split the images among processes
            if self.args.distributed:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                image_names = image_names[rank::world_size]
            else:
                rank = 0

            for image_idx, image_name in enumerate(
                tqdm(image_names, desc=f"Inferring (GPU {rank})", ncols=80)
            ):
                name, _ = os.path.splitext(image_name)

                image_path_full = os.path.join(image_folder, image_name)
                result_path = os.path.join(self.result_dir, f"{name}.png")
                if os.path.exists(result_path):
                    self.logger.info(f"Skipping {name}, already exists.")
                    continue
                image0 = cv2.imread(image_path_full)
                if image0 is None:
                    print(f"Failed to read image {image_path_full}, skipping.")
                    continue
                height, width, _ = image0.shape

                # Get parameters
                size = self.args.view_size  # e.g., 512
                downsample_factors = self.args.downsample_factors  # tuple of 3 ints
                d1, d2, d3 = downsample_factors
                s1 = d1 * size
                s2 = d2 * size
                s3 = d3 * size
                pad_size = (s3 - s1) // 2
                delta23 = (s3 - s2) // 2

                # Pad the image to accommodate larger patches
                image_padded = np.pad(
                    image0,
                    pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    mode="reflect",  # Use 'reflect' or 'constant' padding as needed
                )

                # Initialize output arrays
                ncls = self.args.nclass  # Number of classes
                assembled_output = np.zeros((ncls, height, width), dtype=np.float32)
                count = np.zeros((height, width), dtype=np.float32)

                # Iterate over the image to extract patches
                for i in range(0, height - s1 // 2 + 1, s1 // 2):
                    if i + s1 > height:
                        i = height - s1
                    for j in range(0, width - s1 // 2 + 1, s1 // 2):
                        if j + s1 > width:
                            j = width - s1

                        # Extract patches at different scales
                        img1_patch = image0[i : i + s1, j : j + s1, :]
                        img2_patch = image_padded[
                            i + delta23 : i + delta23 + s2,
                            j + delta23 : j + delta23 + s2,
                            :,
                        ]
                        img3_patch = image_padded[i : i + s3, j : j + s3, :]

                        # Resize patches to the desired size
                        img1_patch_resized = cv2.resize(
                            img1_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        )
                        img2_patch_resized = cv2.resize(
                            img2_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        )
                        img3_patch_resized = cv2.resize(
                            img3_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        )
                        
                        # visualization attention map
                        save_name = None
                        if save_name:
                            # save image
                            if not os.path.exists("./attention_map") :
                                os.makedirs("./attention_map")
                            cv2.imwrite(f"./attention_map/{save_name}_1.png", img1_patch_resized)
                            cv2.imwrite(f"./attention_map/{save_name}_2.png", img2_patch_resized)
                            cv2.imwrite(f"./attention_map/{save_name}_3.png", img3_patch_resized)

                        img1_tensor = normalize_function_mm(img1_patch_resized, self.device)
                        img2_tensor = normalize_function_mm(img2_patch_resized, self.device)
                        img3_tensor = normalize_function_mm(img3_patch_resized, self.device)

                        # Run the model
                        outputs = self.model((img1_tensor, img2_tensor, img3_tensor, save_name))
                        outputs = (
                            outputs.cpu().detach().numpy()
                        )  # Shape: (1, ncls, h_out, w_out)
                        outputs = outputs[0]  # Remove batch dimension

                        # Resize outputs to s1 size if necessary
                        output_resized = cv2.resize(
                            outputs.transpose(1, 2, 0),
                            (s1, s1),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        output_resized = output_resized.transpose(
                            2, 0, 1
                        )  # Shape: (ncls, s1, s1)
                        pred_patch = np.argmax(output_resized, axis=0).astype(np.uint8)
                        pred_patch[pred_patch == 1] = 255
                        if save_name:
                            cv2.imwrite(f"/home/faye/code/linerefactor/attention/{save_name}_pred.png", pred_patch)

                        # Add outputs to assembled_output
                        assembled_output[:, i : i + s1, j : j + s1] += output_resized
                        count[i : i + s1, j : j + s1] += 1

                # Avoid division by zero
                count[count == 0] = 1
                assembled_output /= count

                # Take argmax over classes to get predicted labels
                pred = np.argmax(assembled_output, axis=0).astype(np.uint8)

                # Map class indices to labels if necessary
                # For example, if class 1 is 255 in the mask
                pred[pred == 1] = 255

                # Save the predicted mask
                cv2.imwrite(result_path, pred)

                self.logger.info(f"Saved prediction for {name} at {result_path}")

            self.logger.info("Inference completed successfully.")

            # Synchronize processes if distributed
            if dist.is_initialized():
                dist.barrier()
                
    def predict_whole_sv(self):
        """Performs prediction on input images for single-view models."""
        self.model.eval()
        with torch.no_grad():
            # Ensure synchronization if distributed
            if dist.is_initialized():
                dist.barrier()

            # Get the list of images
            image_names = self.image_names
            image_folder = self.image_folder

            # If distributed, split the images among processes
            if self.args.distributed:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                image_names = image_names[rank::world_size]
            else:
                rank = 0

            for image_idx, image_name in enumerate(
                tqdm(image_names, desc=f"Inferring (GPU {rank})", ncols=80)
            ):
                name, _ = os.path.splitext(image_name)
                image_path_full = os.path.join(image_folder, image_name)
                result_path = os.path.join(self.result_dir, f"{name}.png")
                if os.path.exists(result_path):
                    self.logger.info(f"Skipping {name}, already exists.")
                    continue
                image0 = cv2.imread(image_path_full)
                if image0 is None:
                    print(f"Failed to read image {image_path_full}, skipping.")
                    continue
                height, width, _ = image0.shape

                # Get parameters - Simplified for single view
                size = self.args.view_size  # e.g., 512
                s1 = size # Single view size is just view_size
                # No downsample factors or padding needed for single view

                # Initialize output arrays
                ncls = self.args.nclass  # Number of classes
                assembled_output = np.zeros((ncls, height, width), dtype=np.float32)
                count = np.zeros((height, width), dtype=np.float32)

                # Iterate over the image to extract patches
                for i in range(0, height - s1 // 2 + 1, s1 // 2):
                    if i + s1 > height:
                        i = height - s1
                    for j in range(0, width - s1 // 2 + 1, s1 // 2):
                        if j + s1 > width:
                            j = width - s1

                        # Extract patch at single scale
                        img_patch = image0[i : i + s1, j : j + s1, :]

                        # Resize patch to the desired size - Not necessary as patch is already view_size
                        # img_patch_resized = cv2.resize(
                        #     img_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        # )
                        img_patch_resized = img_patch # No resize needed as patch is already 'size' x 'size'


                        img_tensor = normalize_function_mm(img_patch_resized, self.device) # Normalize single image tensor

                        # Run the model - Pass single image tensor
                        outputs = self.model(img_tensor)
                        outputs = (
                            outputs.cpu().numpy()
                        )  # Shape: (1, ncls, h_out, w_out)
                        outputs = outputs[0]  # Remove batch dimension

                        # Resize outputs to s1 size if necessary - Not necessary as output should be aligned with input patch size if model is FCN
                        output_resized = outputs.transpose(1, 2, 0) # No resize needed if output size is same as input patch size
                        # output_resized = cv2.resize(
                        #     outputs.transpose(1, 2, 0),
                        #     (s1, s1),
                        #     interpolation=cv2.INTER_LINEAR,
                        # )
                        output_resized = output_resized.transpose(
                            2, 0, 1
                        )  # Shape: (ncls, s1, s1)


                        # Add outputs to assembled_output
                        assembled_output[:, i : i + s1, j : j + s1] += output_resized
                        count[i : i + s1, j : j + s1] += 1

                # Avoid division by zero
                count[count == 0] = 1
                assembled_output /= count

                # Take argmax over classes to get predicted labels
                pred = np.argmax(assembled_output, axis=0).astype(np.uint8)

                # Map class indices to labels if necessary
                # For example, if class 1 is 255 in the mask
                pred[pred == 1] = 255

                # Save the predicted mask
                cv2.imwrite(result_path, pred)

                self.logger.info(f"Saved prediction for {name} at {result_path}")

            self.logger.info("Inference completed successfully.")

            # Synchronize processes if distributed
            if dist.is_initialized():
                dist.barrier()

    def predict_whole_sv_with_coords(self):
        """Performs prediction on TIFF images for single-view models using GDAL."""
        self.model.eval()
        with torch.no_grad():
            # Ensure synchronization if distributed
            if dist.is_initialized():
                dist.barrier()

            # Get the list of images
            image_names = self.image_names
            image_folder = self.image_folder

            # If distributed, split the images among processes
            if self.args.distributed:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                image_names = image_names[rank::world_size]
            else:
                rank = 0

            for image_idx, image_name in enumerate(
                tqdm(image_names, desc=f"Inferring (GPU {rank})", ncols=80)
            ):
                name, _ = os.path.splitext(image_name)
                image_path_full = os.path.join(image_folder, image_name)
                result_path = os.path.join(self.result_dir, f"{name}_pred.tif")
                
                if os.path.exists(result_path):
                    self.logger.info(f"Skipping {name}, already exists.")
                    continue

                # Open TIFF with GDAL
                dataset = gdal.Open(image_path_full)
                if dataset is None:
                    print(f"Failed to read TIFF {image_path_full}, skipping.")
                    continue
                    
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                bands = dataset.RasterCount
                
                # Read image data as numpy array (assuming RGB, 3 bands)
                image0 = np.zeros((height, width, bands), dtype=np.uint8)
                for band in range(bands):
                    image0[:, :, band] = dataset.GetRasterBand(band + 1).ReadAsArray()

                # Get geotransform and projection
                geotransform = dataset.GetGeoTransform()
                projection = dataset.GetProjection()

                # Get parameters
                size = self.args.view_size  # e.g., 512
                s1 = size
                ncls = self.args.nclass  # Number of classes

                # Initialize output arrays
                assembled_output = np.zeros((ncls, height, width), dtype=np.float32)
                count = np.zeros((height, width), dtype=np.float32)

                # Iterate over the image to extract patches
                for i in range(0, height - s1 // 2 + 1, s1 // 2):
                    if i + s1 > height:
                        i = height - s1
                    for j in range(0, width - s1 // 2 + 1, s1 // 2):
                        if j + s1 > width:
                            j = width - s1

                        # Extract patch
                        img_patch = image0[i:i + s1, j:j + s1, :]

                        # Prepare tensor
                        img_tensor = normalize_function_mm(img_patch, self.device)

                        # Run the model
                        outputs = self.model(img_tensor)
                        outputs = outputs.cpu().numpy()[0]  # Shape: (ncls, h_out, w_out)

                        # Transpose outputs
                        output_resized = outputs.transpose(1, 2, 0)
                        output_resized = output_resized.transpose(2, 0, 1)

                        # Add outputs to assembled_output
                        assembled_output[:, i:i + s1, j:j + s1] += output_resized
                        count[i:i + s1, j:j + s1] += 1

                # Avoid division by zero
                count[count == 0] = 1
                assembled_output /= count

                # Take argmax to get predicted labels
                pred = np.argmax(assembled_output, axis=0).astype(np.uint8)

                # Map class indices if necessary
                pred[pred == 1] = 255  # Example mapping

                # Save prediction as GeoTIFF
                driver = gdal.GetDriverByName('GTiff')
                out_dataset = driver.Create(
                    result_path,
                    width,
                    height,
                    1,  # Single band for prediction
                    gdal.GDT_Byte  # Assuming 8-bit prediction output
                )
                
                # Set geotransform and projection
                out_dataset.SetGeoTransform(geotransform)
                out_dataset.SetProjection(projection)
                
                # Write prediction data
                out_band = out_dataset.GetRasterBand(1)
                out_band.WriteArray(pred)
                out_band.FlushCache()
                
                # Clean up
                out_dataset = None
                dataset = None

                self.logger.info(f"Saved prediction for {name} at {result_path}")

            self.logger.info("Inference completed successfully.")

            # Synchronize processes if distributed
            if dist.is_initialized():
                dist.barrier()

    def predict_whole_mv_with_coords(self):
        """Performs prediction on TIFF images with multi-view support using GDAL."""
        self.model.eval()
        with torch.no_grad():
            # Ensure synchronization if distributed
            if dist.is_initialized():
                dist.barrier()

            # Get the list of images
            image_names = self.image_names
            image_folder = self.image_folder

            # If distributed, split the images among processes
            if self.args.distributed:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                image_names = image_names[rank::world_size]
            else:
                rank = 0

            for image_idx, image_name in enumerate(
                tqdm(image_names, desc=f"Inferring (GPU {rank})", ncols=80)
            ):
                name, _ = os.path.splitext(image_name)
                image_path_full = os.path.join(image_folder, image_name)
                result_path = os.path.join(self.result_dir, f"{name}_pred.tif")
                
                if os.path.exists(result_path):
                    self.logger.info(f"Skipping {name}, already exists.")
                    continue

                # Open TIFF with GDAL
                dataset = gdal.Open(image_path_full)
                if dataset is None:
                    print(f"Failed to read TIFF {image_path_full}, skipping.")
                    continue
                    
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                bands = dataset.RasterCount
                
                # Read image data as numpy array
                image0 = np.zeros((height, width, bands), dtype=np.uint8)
                for band in range(bands):
                    image0[:, :, band] = dataset.GetRasterBand(band + 1).ReadAsArray()

                # Get geotransform and projection
                geotransform = dataset.GetGeoTransform()
                projection = dataset.GetProjection()

                # Get parameters
                size = self.args.view_size  # e.g., 512
                downsample_factors = self.args.downsample_factors  # tuple of 3 ints
                d1, d2, d3 = downsample_factors
                s1 = d1 * size
                s2 = d2 * size
                s3 = d3 * size
                pad_size = (s3 - s1) // 2
                delta23 = (s3 - s2) // 2

                # Pad the image to accommodate larger patches
                image_padded = np.pad(
                    image0,
                    pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    mode="reflect",
                )

                # Initialize output arrays
                ncls = self.args.nclass  # Number of classes
                assembled_output = np.zeros((ncls, height, width), dtype=np.float32)
                count = np.zeros((height, width), dtype=np.float32)

                # Iterate over the image to extract patches
                for i in range(0, height - s1 // 2 + 1, s1 // 2):
                    if i + s1 > height:
                        i = height - s1
                    for j in range(0, width - s1 // 2 + 1, s1 // 2):
                        if j + s1 > width:
                            j = width - s1

                        # Extract patches at different scales
                        img1_patch = image0[i:i + s1, j:j + s1, :]
                        img2_patch = image_padded[
                            i + delta23:i + delta23 + s2,
                            j + delta23:j + delta23 + s2,
                            :,
                        ]
                        img3_patch = image_padded[i:i + s3, j:j + s3, :]

                        # Resize patches to the desired size
                        img1_patch_resized = cv2.resize(
                            img1_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        )
                        img2_patch_resized = cv2.resize(
                            img2_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        )
                        img3_patch_resized = cv2.resize(
                            img3_patch, (size, size), interpolation=cv2.INTER_LINEAR
                        )

                        # Prepare tensors
                        img1_tensor = normalize_function_mm(img1_patch_resized, self.device)
                        img2_tensor = normalize_function_mm(img2_patch_resized, self.device)
                        img3_tensor = normalize_function_mm(img3_patch_resized, self.device)

                        # Run the model
                        outputs = self.model((img1_tensor, img2_tensor, img3_tensor))
                        outputs = outputs.cpu().numpy()[0]  # Shape: (ncls, h_out, w_out)

                        # Resize outputs
                        output_resized = cv2.resize(
                            outputs.transpose(1, 2, 0),
                            (s1, s1),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        output_resized = output_resized.transpose(2, 0, 1)

                        # Add outputs to assembled_output
                        assembled_output[:, i:i + s1, j:j + s1] += output_resized
                        count[i:i + s1, j:j + s1] += 1

                # Avoid division by zero
                count[count == 0] = 1
                assembled_output /= count

                # Take argmax to get predicted labels
                pred = np.argmax(assembled_output, axis=0).astype(np.uint8)

                # Map class indices if necessary
                pred[pred == 1] = 255  # Example mapping

                # Save prediction as GeoTIFF
                driver = gdal.GetDriverByName('GTiff')
                out_dataset = driver.Create(
                    result_path,
                    width,
                    height,
                    1,  # Single band for prediction
                    gdal.GDT_Byte  # Adjust if needed
                )
                
                # Set geotransform and projection
                out_dataset.SetGeoTransform(geotransform)
                out_dataset.SetProjection(projection)
                
                # Write prediction data
                out_band = out_dataset.GetRasterBand(1)
                out_band.WriteArray(pred)
                out_band.FlushCache()
                
                # Clean up
                out_dataset = None
                dataset = None

                self.logger.info(f"Saved prediction for {name} at {result_path}")

            self.logger.info("Inference completed successfully.")

            # Synchronize processes if distributed
            if dist.is_initialized():
                dist.barrier()
                
