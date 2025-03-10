import os
import cv2
import logging
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from .base import BaseEngine
from .utils import infer_visual_one_whole_image
from ..datasets import WHUBuildingVector, GLHWaterVector, VHRRoadVector
from ..utils.loss import NormalLossWithMask, AngleLossWithMask
from ..utils.metrics import coco_eval, polis_eval


class TrainEngine(BaseEngine):
    def __init__(self, args):
        super(TrainEngine, self).__init__(args)

        self.global_step = 0  # Initialize global step counter
        args_dir = str(args.corner_angle_threshold) + "_" + str(args.d) + "_" + str(args.num_points)
        self.run_dir = os.path.join(
            os.getcwd(), self.args.run_dir, self.args.dataset, self.args.model, "train", args_dir
        )
        os.makedirs(self.run_dir, exist_ok=True)

        if self.args.rank == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            logging.basicConfig(
                filename=os.path.join(self.run_dir, "training.log"),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
            self.writer = SummaryWriter(
                log_dir=self.run_dir
            )  # Step 2: Initialize SummaryWriter
        else:
            logging.basicConfig(level=logging.WARNING)
            self.writer = None  # Only main process writes logs

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

        # 数据加载器
        self.train_loader, self.val_loader = self._build_data()
        self._build_eval_data()

        # 模型、损失函数、优化器、学习率调度器
        if self.args.loss_type == "normal":
            self.criterion = NormalLossWithMask(
                weight_regression=self.args.weight_regression,
                weight_classification=self.args.weight_classification,
            ).to(self.device)

        elif self.args.loss_type == "angle":
            self.criterion = AngleLossWithMask(
                weight_regression=self.args.weight_regression,
                weight_classification=self.args.weight_classification,
                weight_angle=self.args.weight_angle,
                corner_threshold=self.args.corner_angle_threshold,
                noncorner_threshold=self.args.noncorner_angle_threshold,
            ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learn_rate)
        if self.args.eval_whole:
            # coco ap metrics
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5
            )
        # 其他
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.best_ap = 0.0
        self.best_model_path = os.path.join(self.run_dir, "best_model.pth")
        self.last_model_path = os.path.join(self.run_dir, "last_model.pth")
        # 如果需要恢复训练
        if self.args.resume is not None:
            self._load_checkpoint()

        if args.distributed:
            self.model = DDP(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank
            )

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume, map_location=self.device)
        self.start_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info(
            f"Loaded checkpoint from {self.args.resume} at epoch {self.start_epoch}"
        )

    def _build_data(self):
        # 数据集路径
        train_h5_path = self.args.train_h5_path
        if not self.args.eval_whole:
            val_h5_path = self.args.val_h5_path
        # 数据集
        # 定义预处理, 将图像缩放到 512x512，归一化，转换为张量
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if self.args.dataset == "glhwater":
            Dataset = GLHWaterVector
        elif self.args.dataset == "whubuilding":
            Dataset = WHUBuildingVector
        elif self.args.dataset == "vhrroad":
            Dataset = VHRRoadVector
        else:
            raise ValueError(f"Invalid dataset: {self.args.dataset}")
        train_dataset = Dataset(
            train_h5_path, self.args.down_ratio, transform
        )
        if not self.args.eval_whole:
            val_dataset = Dataset(
                val_h5_path, self.args.down_ratio, transform
            )
        if self.args.distributed:
            # 分布式采样器
            train_sampler = DistributedSampler(train_dataset)
            if not self.args.eval_whole:
                val_sampler = DistributedSampler(val_dataset)
            # 数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=train_sampler,
            )
            if not self.args.eval_whole:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    sampler=val_sampler,
                )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
            if not self.args.eval_whole:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                )

        if self.args.rank == 0:
            print(f"Train data length: {len(train_dataset)}")
            print(f"Train loader length: {len(train_loader)}")
            if not self.args.eval_whole:
                print(f"Val data length: {len(val_dataset)}")
                print(f"Val loader length: {len(val_loader)}")

        if not self.args.eval_whole:
            return train_loader, val_loader
        else:
            return train_loader, None

    def _build_eval_data(self):
        """Builds the list of images and loads predicted points."""
        # Get the image folder path from arguments
        image_folder = self.args.image_path
        self.val_image_folder = os.path.join(image_folder, "val", "img")
        self.test_image_folder = os.path.join(image_folder, "test", "img")

        # Create the list of images
        val_image_names = os.listdir(self.val_image_folder)
        test_image_names = os.listdir(self.test_image_folder)
        # Filter for image files
        val_image_names = [
            f for f in val_image_names if f.endswith(".jpg") and not f.startswith(".")
        ]
        test_image_names = [
            f for f in test_image_names if f.endswith(".jpg") and not f.startswith(".")
        ]
        self.val_image_names = sorted(val_image_names)
        self.test_image_names = sorted(test_image_names)

        val_coco_predictions_path = os.path.join(
            image_folder, "val", "predict", "swin_l", "swinl_upernet_nofliter.json"
        )
        test_coco_predictions_path = os.path.join(
            image_folder, "test", "predict", "swin_l", "swinl_upernet_nofliter.json"
        )
        self.val_coco_labels_path = os.path.join(
            image_folder, "val", "coco_label_with_inter.json"
        )
        self.test_coco_labels_path = os.path.join(
            image_folder, "test", "coco_label_with_inter.json"
        )

        # Load COCO-format predictions
        with open(val_coco_predictions_path, "r") as f:
            self.val_coco_predictions = json.load(f)
        with open(test_coco_predictions_path, "r") as f:
            self.test_coco_predictions = json.load(f)

        # Organize predictions by image_id
        self.val_predictions_by_image = {}
        for ann in self.val_coco_predictions["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.val_predictions_by_image:
                self.val_predictions_by_image[image_id] = []
            self.val_predictions_by_image[image_id].append(ann)

        self.test_coco_predictions_by_image = {}
        for ann in self.test_coco_predictions["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.test_coco_predictions_by_image:
                self.test_coco_predictions_by_image[image_id] = []
            self.test_coco_predictions_by_image[image_id].append(ann)

        # Map image filenames to image_ids
        self.val_image_id_map = {}
        for img in self.val_coco_predictions["images"]:
            self.val_image_id_map[img["file_name"]] = img["id"]

        self.test_image_id_map = {}
        for img in self.test_coco_predictions["images"]:
            self.test_image_id_map[img["file_name"]] = img["id"]

        if self.args.rank == 0:
            print(f"Val image length: {len(self.val_image_names)}")
            print(f"Test image length: {len(self.test_image_names)}")

    def train(self):
        num_epochs = self.args.epochs
        # self.validate(0)
        self._eval_whole(0, eval_type="all", split="val")
        for epoch in range(self.start_epoch, num_epochs):
            # 设置 sampler 的 epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                if not self.args.eval_whole:
                    self.val_loader.sampler.set_epoch(epoch)
            if self.args.rank == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
            # 训练一个 epoch
            train_loss = self._train_one_epoch(epoch)
            if self.args.rank == 0 and self.writer:
                self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            # 验证
            if self.args.eval_whole and epoch + 1 >= 5:
                coco_eval = None
                ap = None

                coco_eval = self._eval_whole(epoch, eval_type="all", split="val")

                if self.args.rank == 0:
                    ap = coco_eval[0]

                if self.args.distributed:
                    # 广播 ap
                    ap_tensor = torch.tensor(
                        [ap if self.args.rank == 0 else 0.0],
                        dtype=torch.float32,
                        device=self.args.device,
                    )
                    dist.broadcast(ap_tensor, src=0)
                    ap = ap_tensor.item()

                self.scheduler.step(ap)

                if ap > self.best_ap:
                    self.best_ap = ap
                    if self.args.rank == 0:
                        # Save the best model.
                        if self.args.distributed:
                            torch.save(
                                self.model.module.state_dict(), self.best_model_path
                            )
                        else:
                            torch.save(self.model.state_dict(), self.best_model_path)

                        self.logger.info(f"Saved best model at epoch {epoch + 1}")

                    if self.args.distributed:
                        torch.distributed.barrier()

                    self._eval_whole(epoch, eval_type="all", split="test")

                if self.args.rank == 0:
                    self.save_checkpoint(epoch, is_best=False)
                    
            # else:
            #     val_loss = self.validate(epoch)
            #     # 学习率调度
            #     self.scheduler.step(val_loss)

            #     # 保存模型
            #     if self.args.rank == 0:
            #         if val_loss < self.best_loss:
            #             self.best_loss = val_loss
            #             # Save the best model.
            #             if self.args.distributed:
            #                 torch.save(
            #                     self.model.module.state_dict(), self.best_model_path
            #                 )
            #             else:
            #                 torch.save(self.model.state_dict(), self.best_model_path)

            #             self.logger.info(f"Saved best model at epoch {epoch + 1}")
            #         # 保存最新模型
            #         self.save_checkpoint(epoch, is_best=False)
            #         # Step 3: Log metrics to TensorBoard (Epoch level)
            #         if self.writer:
            #             self.writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            # 同步所有进程
            if self.args.distributed:
                torch.distributed.barrier()
        if self.args.rank == 0:
            self.logger.info("training complete.")
        # 加载最佳模型
        model_state_dict = torch.load(self.best_model_path)
        if self.args.distributed:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        # TODO:all 会报错，不知道什么原因
        self._eval_whole(num_epochs, eval_type="all", split="test")
        if self.writer:
            self.writer.close()

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        if self.args.loss_type == "angle":
            total_angle_loss = 0.0
        if self.args.rank == 0:
            tbar = tqdm(self.train_loader, desc="Training", ncols=80)
        else:
            tbar = self.train_loader
        for batch in tbar:
            # 获取数据
            image = batch["image"].to(self.device, non_blocking=True)
            pred_points = batch["pred_points"].to(self.device, non_blocking=True)
            gt_points = batch["gt_points"].to(self.device, non_blocking=True)
            is_corner_gt = batch["is_corner"].to(self.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
            # 前向传播
            refined_points, is_corner_logits = self.model(
                image, pred_points, valid_mask
            )
            # 计算损失
            if self.args.loss_type == "normal":
                loss, reg_loss, cls_loss = self.criterion(
                    refined_points,
                    gt_points,
                    is_corner_logits,
                    is_corner_gt,
                    valid_mask,
                )
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                # 记录损失
                total_loss += loss.item()
                total_reg_loss += reg_loss
                total_cls_loss += cls_loss

                # Step 3: Log per-iteration training loss to TensorBoard
                if self.args.rank == 0 and self.writer:
                    self.writer.add_scalar(
                        "Loss/train_iter", loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/train_reg_iter", reg_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/train_cls_iter", cls_loss, self.global_step
                    )
                    self.global_step += 1  # Increment global step
            elif self.args.loss_type == "angle":
                loss, reg_loss, cls_loss, angle_loss = self.criterion(
                    refined_points,
                    gt_points,
                    is_corner_logits,
                    is_corner_gt,
                    valid_mask,
                )
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                # 记录损失
                total_loss += loss.item()
                total_reg_loss += reg_loss
                total_cls_loss += cls_loss
                total_angle_loss += angle_loss

                # Step 3: Log per-iteration training loss to TensorBoard
                if self.args.rank == 0 and self.writer:
                    self.writer.add_scalar(
                        "Loss/train_iter", loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/train_reg_iter", reg_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/train_cls_iter", cls_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/train_angle_iter", angle_loss, self.global_step
                    )
                    self.global_step += 1
            else:
                raise ValueError(f"Invalid loss type: {self.args.loss_type}")

            # 更新进度条
            if self.args.rank == 0:
                tbar.set_postfix(loss=loss.item())
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        if self.args.rank == 0:
            self.logger.info(
                f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f} (Reg: {avg_reg_loss:.4f}, Cls: {avg_cls_loss:.4f})"
            )
            # Step 3: Log epoch-level training loss to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train_avg", avg_loss, epoch)
                self.writer.add_scalar("Loss/train_reg", avg_reg_loss, epoch)
                self.writer.add_scalar("Loss/train_cls", avg_cls_loss, epoch)
        return avg_loss

    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        if self.args.rank == 0:
            tbar = tqdm(self.val_loader, desc="Validation", ncols=80)
        else:
            tbar = self.val_loader
        with torch.no_grad():
            for batch in tbar:
                # 获取数据
                image = batch["image"].to(self.device, non_blocking=True)
                idx = batch["idx"]
                pred_points = batch["pred_points"].to(self.device, non_blocking=True)
                gt_points = batch["gt_points"].to(self.device, non_blocking=True)
                is_corner_gt = batch["is_corner"].to(self.device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
                # 前向传播
                refined_points, is_corner_logits = self.model(
                    image, pred_points, valid_mask
                )
                # 计算损失
                loss, reg_loss, cls_loss = self.criterion(
                    refined_points,
                    gt_points,
                    is_corner_logits,
                    is_corner_gt,
                    valid_mask,
                )
                # 记录损失
                total_loss += loss.item()
                total_reg_loss += reg_loss
                total_cls_loss += cls_loss

                # Step 3: Log per-iteration validation loss to TensorBoard
                if self.args.rank == 0 and self.writer:
                    self.writer.add_scalar(
                        "Loss/val_iter", loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/val_reg_iter", reg_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/val_cls_iter", cls_loss, self.global_step
                    )

                # 更新进度条
                if self.args.rank == 0:
                    tbar.set_postfix(loss=loss.item())

                # save the original image&points and the refined image&gt_points
                if epoch % 10 == 1 or epoch == 0:
                    ori_image_wh = batch["ori_image_wh"]
                    for i in range(len(ori_image_wh[0])):
                        if idx[i] % 100 == 0:
                            ori_image = (
                                batch["image"][i].cpu().numpy().transpose(1, 2, 0)
                            )
                            ori_image = (
                                ori_image * np.array([0.229, 0.224, 0.225])
                                + np.array([0.485, 0.456, 0.406])
                            ) * 255
                            ori_image = cv2.resize(
                                ori_image,
                                (ori_image_wh[1][i].item(), ori_image_wh[0][i].item()),
                            )
                            ori_image = ori_image.astype(np.uint8)
                            self._save_image_and_points(
                                epoch,
                                idx[i],
                                ori_image,
                                pred_points[i],
                                refined_points[i],
                                gt_points[i],
                                is_corner_logits[i],
                                is_corner_gt[i],
                            )

        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_reg_loss = total_reg_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)
        # 所有进程计算损失总和
        avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
        avg_reg_loss_tensor = torch.tensor([avg_reg_loss], device=self.device)
        avg_cls_loss_tensor = torch.tensor([avg_cls_loss], device=self.device)
        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            avg_reg_loss_tensor, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            avg_cls_loss_tensor, op=torch.distributed.ReduceOp.SUM
        )
        avg_loss = avg_loss_tensor.item() / self.args.world_size
        avg_reg_loss = avg_reg_loss_tensor.item() / self.args.world_size
        avg_cls_loss = avg_cls_loss_tensor.item() / self.args.world_size
        if self.args.rank == 0:
            self.logger.info(
                f"Epoch {epoch + 1} Validation Loss: {avg_loss:.4f} (Reg: {avg_reg_loss:.4f}, Cls: {avg_cls_loss:.4f})"
            )
            # Step 3: Log epoch-level validation loss to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/val_avg", avg_loss, epoch)
                self.writer.add_scalar("Loss/val_reg", avg_reg_loss, epoch)
                self.writer.add_scalar("Loss/val_cls", avg_cls_loss, epoch)
        return avg_loss

    def _save_image_and_points(
        self,
        epoch,
        idx,
        ori_image,
        pred_points,
        refined_points,
        gt_points,
        is_corner_logits,
        is_corner_gt,
    ):
        save_dir = os.path.join(self.run_dir, "save_images", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        # 保存image+gt_points, 从512*512恢复到原始尺寸
        image = ori_image.copy()
        ori_w, ori_h = image.shape[1], image.shape[0]
        gt_points = gt_points.cpu().numpy()
        gt_points = gt_points * np.array([ori_w, ori_h]) / 512
        for x, y in gt_points:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imwrite(
            os.path.join(save_dir, f"{idx}_gt.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        # 保存image+pred_points
        image = ori_image.copy()
        pred_points = pred_points.cpu().numpy()
        pred_points = (
            pred_points * np.array([ori_w, ori_h]) * self.args.down_ratio / 512
        )
        for x, y in pred_points:
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.imwrite(
            os.path.join(save_dir, f"{idx}_pred.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        # 保存image+refined_points
        image = ori_image.copy()
        refined_points = refined_points.cpu().numpy()
        refined_points = refined_points * np.array([ori_w, ori_h]) / 512
        for x, y in refined_points:
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imwrite(
            os.path.join(save_dir, f"{idx}_refined.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch + 1,
            "model_state_dict": (
                self.model.module.state_dict()
                if self.args.distributed
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
        }
        torch.save(state, self.last_model_path)

    def _eval_whole(self, epoch, eval_type="coco", split="val"):
        """对输入图像执行推理。"""
        """只有rank=0的进程会执行评估代码"""
        self.model.eval()
        with torch.no_grad():
            # 如果是分布式环境，确保同步
            if dist.is_initialized():
                dist.barrier()

            # 获取图像列表
            if split == "val":
                image_names = self.val_image_names
                image_folder = self.val_image_folder
                image_id_map = self.val_image_id_map
                predictions_by_image = self.val_predictions_by_image
                label_path = self.val_coco_labels_path
            elif split == "test":
                image_names = self.test_image_names
                image_folder = self.test_image_folder
                image_id_map = self.test_image_id_map
                predictions_by_image = self.test_coco_predictions_by_image
                label_path = self.test_coco_labels_path
            else:
                raise ValueError(f"Invalid split: {split}")

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
                image_id = image_id_map[image_name]
                predictions = predictions_by_image.get(image_id, [])
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
                )
                refined_annotations.extend(one_results)

            if self.args.distributed:
                # 创建一个列表用于存储所有进程的结果
                output_objects = [None for _ in range(dist.get_world_size())]

                # 收集所有进程的结果
                dist.barrier()
                dist.all_gather_object(output_objects, refined_annotations)

                # 在主进程(rank 0)合并结果
                if self.args.rank == 0:
                    # 展平列表
                    all_results = [
                        item for sublist in output_objects for item in sublist
                    ]
                else:
                    all_results = []
            else:
                all_results = refined_annotations

        if self.args.rank == 0:
            num_process = 30

            # make coco dict for metric
            coco_dict = {
                "annotations": all_results,
            }
            print(f"all_results: {len(all_results)}")

            # 评估结果
            if eval_type == "coco":
                if len(all_results) == 0:
                    coco_metric = [0, 0, 0, 0, 0]
                else:
                    coco_metric = coco_eval(
                        annFile=label_path,
                        resFile=coco_dict,
                        num_processes=num_process,
                    )
                self.logger.info(
                    f"Epoch {epoch + 1} {split} COCO Metric: {coco_metric}"
                )
            elif eval_type == "all":
                if len(all_results) == 0:
                    coco_metric = [0, 0, 0, 0, 0]
                    # f1, iou, ciou, biou = 0.0, 0.0, 0.0, 0.0
                    polis = np.nan
                else:
                    polis = polis_eval(
                        annFile=label_path,
                        resFile=coco_dict.copy(),
                        num_processes=num_process,
                    )
                    coco_metric = coco_eval(
                        annFile=label_path,
                        resFile=coco_dict.copy(),
                        num_processes=num_process,
                    )
                    # f1, iou, ciou, biou = compute_IoU_cIoU_bIoU(
                    #     input_json=coco_dict,
                    #     gti_annotations=label_path,
                    #     num_processes=num_process,
                    # )
                self.logger.info(
                    f"Epoch {epoch + 1} {split} COCO Metric: {coco_metric}"
                )
                # self.logger.info(f"Epoch {epoch + 1} {split} F1: {f1}")
                # self.logger.info(f"Epoch {epoch + 1} {split} IoU: {iou}")
                # self.logger.info(f"Epoch {epoch + 1} {split} CIoU: {ciou}")
                # self.logger.info(f"Epoch {epoch + 1} {split} BIoU: {biou}")
                self.logger.info(f"Epoch {epoch + 1} {split} Polis: {polis}")
            else:
                raise ValueError(f"Invalid eval_type: {eval_type}")

            return coco_metric
