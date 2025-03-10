import logging
import os
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .base import BaseEngine
from .utils import infer_visual_one_whole_image
from ..utils.metrics import coco_eval, compute_IoU_cIoU_bIoU, polis_eval

class InferImageEngine(BaseEngine):
    def __init__(self, args):
        """Initialize the inference engine.

        Args:
            args: Arguments containing configuration parameters.
        """
        super(InferImageEngine, self).__init__(args)
        self.global_step = 0  # Initialize global step counter
        args_dir = str(args.corner_angle_threshold) + "_" + str(args.d) + "_" + str(args.num_points) + "_" + str(args.corner_threshold)
        
        self.run_dir = os.path.join(
            os.getcwd(), self.args.run_dir, self.args.dataset, self.args.model, "infer", args_dir
        )

        os.makedirs(self.run_dir, exist_ok=True)
        # Setup logging
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

        # Build the data list for inference
        self._build_data()

        if self.args.resume is not None:
            self.load_resume()

        if args.distributed:
            self.model = DDP(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank
            )

    def load_resume(self):
        self.model.load_state_dict(torch.load(self.args.resume))
        self.logger.info(f"Loaded the model from {self.args.resume}")

    def _build_data(self):
        """Builds the list of images and loads predicted points."""
        # Get the image folder path from arguments
        image_folder = self.args.image_path
        self.image_folder = image_folder

        # Create the list of images
        image_names = os.listdir(image_folder)
        # Filter for image files
        image_names = [
            f for f in image_names if f.endswith(".jpg") and not f.startswith(".")
        ]
        self.image_names = sorted(image_names)

        # Load COCO-format predictions
        with open(self.args.coco_predictions, "r") as f:
            self.coco_predictions = json.load(f)

        # Organize predictions by image_id
        self.predictions_by_image = {}
        for ann in self.coco_predictions["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.predictions_by_image:
                self.predictions_by_image[image_id] = []
            self.predictions_by_image[image_id].append(ann)

        # Map image filenames to image_ids
        self.image_id_map = {}
        for img in self.coco_predictions["images"]:
            self.image_id_map[img["file_name"]] = img["id"]

        # Prepare the result directory
        self.result_json = self.args.result_json

        # load coco label data for visualization
        with open(self.args.coco_labels, "r") as f:
            self.coco_labels = json.load(f)
        self.labels_by_image = {}
        for ann in self.coco_labels["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.labels_by_image:
                self.labels_by_image[image_id] = []
            self.labels_by_image[image_id].append(ann)

    def predict(self):
        """对输入图像执行推理。"""
        self.model.eval()
        with torch.no_grad():
            # 如果是分布式环境，确保同步
            if dist.is_initialized():
                dist.barrier()

            # 获取图像列表
            image_names = self.image_names
            image_folder = self.image_folder

            # 如果是分布式，分割图像列表
            if self.args.distributed:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                image_names = image_names[rank::world_size]
            else:
                rank = 0

            refined_annotations = []
            for image_idx, image_name in enumerate(
                tqdm(image_names, desc=f"Inferring (GPU {rank})", ncols=80)
            ):
                # 推理
                try:
                    image_id = self.image_id_map[image_name]
                    if image_id not in self.predictions_by_image:
                        continue
                    predictions = self.predictions_by_image[image_id]

                    labels = self.labels_by_image.get(image_id, [])

                    if image_id not in self.labels_by_image:
                        labels = []
                    one_results = infer_visual_one_whole_image(
                        self.model,
                        image_folder,
                        image_name,
                        image_id,
                        predictions,
                        self.args.num_points,
                        self.args.down_ratio,
                        self.args.corner_threshold,
                        self.args.d,
                        self.device,
                        os.path.join(self.run_dir, "infer_visual"),
                        self.args.visual,
                        labels,
                    )
                except Exception as e:
                    # 显示哪一行出错,并写入日志
                    import traceback

                    self.logger.error(
                        f"Error occurred while processing image {image_name}: {e}"
                    )
                    self.logger.error(traceback.format_exc())
                    one_results = []
                refined_annotations.extend(one_results)

            if self.args.distributed:
                # 创建一个列表用于存储所有进程的结果
                output_objects = [None for _ in range(dist.get_world_size())]

                # 收集所有进程的结果
                dist.barrier()
                dist.all_gather_object(output_objects, refined_annotations)

                # 在主进程(rank 0)合并结果
                if dist.get_rank() == 0:
                    # 展平列表
                    all_results = [
                        item for sublist in output_objects for item in sublist
                    ]
                else:
                    all_results = []
            else:
                all_results = refined_annotations

            # 保存结果
            if rank == 0:
                # 保存结果
                coco_dict = {
                    "images": self.coco_predictions["images"],
                    "annotations": all_results,
                    "categories": self.coco_predictions["categories"],
                }
                with open(self.result_json, "w") as f:
                    json.dump(coco_dict, f, indent=4)
                self.logger.info(f"Saved refined annotations to {self.result_json}")

    def metric(self, eval_type, num_process):
        if eval_type == "coco":
            coco_metric = coco_eval(
                annFile=self.args.coco_labels,
                resFile=self.args.result_json,
                num_processes=num_process,
            )
            self.logger.info(f"COCO metric: {coco_metric}")
        elif eval_type == "all":
            coco_metric = coco_eval(
                annFile=self.args.coco_labels,
                resFile=self.args.result_json,
                num_processes=num_process,
            )
            polis = polis_eval(
                annFile=self.args.coco_labels,
                resFile=self.args.result_json,
                num_processes=num_process,
            )
            f1, iou, ciou, biou = compute_IoU_cIoU_bIoU(
                input_json=self.args.result_json,
                gti_annotations=self.args.coco_labels,
                num_processes=num_process,
            )
            self.logger.info(f"COCO metric: {coco_metric}")
            self.logger.info(f"F1: {f1}")
            self.logger.info(f"IoU: {iou}")
            self.logger.info(f"cIoU: {ciou}")
            self.logger.info(f"bIoU: {biou}")
            self.logger.info(f"Polis: {polis}")
        else:
            raise ValueError(f"Unknown eval type: {eval_type}")