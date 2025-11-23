import logging
import os

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .base import BaseEngine
from ..datasets import *
from ..models import UPerNet
from ..utils.loss import SegmentationLosses
from ..utils.metrics import Evaluator
from ..utils.trick import EarlyStopping


class TrainEngine(BaseEngine):
    def __init__(self, args):
        """Initialize the segmentation engine.

        Args:
            args: Arguments containing configuration parameters.
        """
        super(TrainEngine, self).__init__(args)

        # Setup logging.
        logging.basicConfig(
            filename=os.path.join(self.run_dir, "training.log"),
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

        # training related
        self.model = self._build_model().to(self.device)
        if args.distributed:
            self.model = DDP(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank
            )
            
        if self.args.resume is not None:
            self.load_resume()
        
        self.start_epoch = 0
        self.optimizer, self.scheduler = self._build_optimizer()
        self.criterion = self._build_loss()
        self.train_loader, self.val_loader = self._build_data()
        self.evaluator, self.earlystopping = self._build_evaluator()
        self.best_model_path = os.path.join(self.run_dir, "best_model.pth")
        self.last_model_path = os.path.join(self.run_dir, "last_model.pth")

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
                pretrain=True
            )
        else:
            raise ValueError(f"Invalid segmentation head: {self.args.segHead}")

        return model

    def load_resume(self):
        resume_if = torch.load(self.args.resume, map_location=self.device)
        self.start_epoch = resume_if["epoch"]
        self.model.load_state_dict(resume_if["model"])
        self.optimizer.load_state_dict(resume_if["optimizer_state_dict"])
        self.scheduler.load_state_dict(resume_if["scheduler_state_dict"])
        self.earlystopping.criterion = resume_if["earlystopping_criterion"]
        self.logger.info(f"Loaded the model from {self.args.resume}")

    def _build_optimizer(self):
        """Builds the optimizer and scheduler.

        Returns:
            A tuple containing the optimizer and the scheduler.
        """
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.learn_rate, weight_decay=0.05
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=self.args.lr_patience, verbose=True
        )
        return optimizer, scheduler

    def _build_evaluator(self):
        """Builds the evaluator and early stopping mechanism.

        Returns:
            A tuple containing the evaluator and early stopping instance.
        """
        evaluator = Evaluator(
            num_class=self.args.nclass,
            ignore_index=self.args.ignore_index,
            device=self.device,
        )
        earlystopping = EarlyStopping(
            criterion=0, patience=self.args.early_stopping_patience
        )
        return evaluator, earlystopping

    def _build_loss(self):
        """Builds the loss function.

        Returns:
            The loss criterion.
        """
        criterion = SegmentationLosses(
            ignore_index=self.args.ignore_index, device=self.device
        ).build_loss(mode="ce")
        return criterion

    def _build_data(self):
        """Builds the training and validation data loaders.

        Returns:
            A tuple containing the training and validation data loaders.
        """
        train_h5_name = f"{self.args.dataset}_seg_train.h5"
        val_h5_name = f"{self.args.dataset}_seg_val.h5"
        train_h5_path = os.path.join(self.args.data_root, train_h5_name)
        val_h5_path = os.path.join(self.args.data_root, val_h5_name)

        if self.args.dataset == "whubuilding" or self.args.dataset == "farmland":
            if self.args.isContext:
                train_data = WHUBuildingMVTrain(hdf5_path=train_h5_path, training=True)
            else:
                train_data = WHUBuildingSVTrain(hdf5_path=train_h5_path, training=True)
        elif self.args.dataset == "glhwater":
            if self.args.isContext:
                train_data = GLHWaterMVTrain(hdf5_path=train_h5_path, training=True)
            else:
                train_data = GLHWaterSVTrain(hdf5_path=train_h5_path, training=True)
        elif self.args.dataset == "vhrroad":
            if self.args.isContext:
                train_data = VHRRoadMVTrain(hdf5_path=train_h5_path, training=True)
            else:
                train_data = VHRRoadSVTrain(hdf5_path=train_h5_path, training=True)
        # 截取数据集的一部分用于调试
        # train_data = torch.utils.data.Subset(train_data, list(range(500)))

        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                sampler=train_sampler,
                pin_memory=True,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )

        if self.args.dataset == "whubuilding" or self.args.dataset == "farmland":
            if self.args.isContext:
                val_data = WHUBuildingMVTrain(hdf5_path=val_h5_path)
            else:
                val_data = WHUBuildingSVTrain(hdf5_path=val_h5_path)
        elif self.args.dataset == "glhwater":
            if self.args.isContext: 
                val_data = GLHWaterMVTrain(hdf5_path=val_h5_path)
            else:
                val_data = GLHWaterSVTrain(hdf5_path=val_h5_path)
        elif self.args.dataset == "vhrroad":
            if self.args.isContext:
                val_data = VHRRoadMVTrain(hdf5_path=val_h5_path)
            else:
                val_data = VHRRoadSVTrain(hdf5_path=val_h5_path)
        # 截取数据集的一部分用于调试
        # val_data = torch.utils.data.Subset(val_data, list(range(100)))

        if self.args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_data, shuffle=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                sampler=val_sampler,
                pin_memory=True,
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )

        if self.args.rank == 0:
            print(f"Train data length: {len(train_data)}")
            print(f"Train loader length: {len(train_loader)}")
            print(f"Val data length: {len(val_data)}")
            print(f"Val loader length: {len(val_loader)}")

        return train_loader, val_loader

    def train(self):
        """Trains the model."""
        max_epochs = self.args.epochs
        for epoch in range(self.start_epoch, max_epochs + 1):
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch [{epoch}/{max_epochs}]. Learning rate: {lr}/ GPU: {self.args.rank}"
            )
            print(
                f"Epoch [{epoch}/{max_epochs}] Learning rate: {lr}/ GPU: {self.args.rank}"
            )

            if self.args.distributed:
                # wait print
                torch.distributed.barrier()
                # Set the random seed to ensure consistent data order across processes.
                self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            epoch_losses = []
            save_interval = max(
                len(self.train_loader) // 10, 1
            )  # Save model 10 times per epoch.
            tbar = tqdm(
                enumerate(self.train_loader, 1),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
                ncols=80,
            )
            for idx, batch in tbar:
                if self.args.isContext:
                    # Get data and transfer to device.
                    img1 = batch["img1"].to(self.device, non_blocking=True)
                    img2 = batch["img2"].to(self.device, non_blocking=True)
                    img3 = batch["img3"].to(self.device, non_blocking=True)
                    mask = batch["mask"].to(self.device, non_blocking=True)

                    # Forward pass.
                    outputs = self.model((img1, img2, img3, None))
                else:
                    img = batch["img"].to(self.device, non_blocking=True)
                    mask = batch["mask"].to(self.device, non_blocking=True)

                    outputs = self.model(img)
                # Compute loss.
                loss_dict = self.criterion(outputs, mask)
                loss = loss_dict["loss_tot"]
                epoch_losses.append(loss.item())
                # Backward pass and optimization.
                self.model.zero_grad()
                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.args.rank == 0:
                    # Update progress bar with loss.
                    tbar.set_postfix(loss=loss.item())

                # Save the model at save intervals.
                if idx % save_interval == 0 and self.args.rank == 0:
                    self.logger.info(
                        f"Epoch {epoch} Iteration {idx} Loss: {loss.item()}"
                    )

            avg_loss = sum(epoch_losses) / len(epoch_losses)

            if self.args.rank == 0:
                self.logger.info(f"Epoch {epoch} Average Loss: {avg_loss}")

            # Validate the model.
            self.validate(epoch)

            # Update the learning rate scheduler.
            self.scheduler.step(self.earlystopping.criterion)

            # Check for early stopping.
            if self.args.distributed:
                torch.distributed.barrier()
            if self.earlystopping.EarlyStopping:
                print("Early stopping")
                break

            # Save the last model (only in the main process).
            if self.args.rank == 0:
                last_save_if = {
                    "epoch": epoch + 1,
                    "model": (
                        self.model.module.state_dict()
                        if self.args.distributed
                        else self.model.state_dict()
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "earlystopping_criterion": self.earlystopping.criterion,
                }
                torch.save(last_save_if, self.last_model_path)
                self.logger.info(f"Saved the last model to {self.last_model_path}")

    def validate(self, epoch):
        """Validates the model.

        Args:
            epoch: The current epoch number.
        """
        self.model.eval()
        self.evaluator.reset()

        if self.args.distributed:
            # Set the epoch for validation sampler
            self.val_loader.sampler.set_epoch(epoch)

        tbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="Validation",
            ncols=80,
        )
        with torch.no_grad():
            for idx, batch in tbar:
                if self.args.isContext:
                    img1 = batch["img1"].to(self.device, non_blocking=True)
                    img2 = batch["img2"].to(self.device, non_blocking=True)
                    img3 = batch["img3"].to(self.device, non_blocking=True)
                    gt = batch["mask"].to(self.device, non_blocking=True)

                    outputs = self.model((img1, img2, img3, None))
                else:
                    img = batch["img"].to(self.device, non_blocking=True)
                    gt = batch["mask"].to(self.device, non_blocking=True)

                    outputs = self.model(img)
                # Process outputs to get predictions.
                pred = outputs.argmax(1).squeeze(1)
                # Update evaluator.
                for i in range(len(gt)):
                    self.evaluator.add_batch(gt[i], pred[i])

                # Update progress bar with current MIoU.
                if self.args.rank == 0:
                    current_miou = self.evaluator.Mean_Intersection_over_Union()
                    tbar.set_postfix(MIoU=f"{current_miou:.4f}")

        # Synchronize evaluation results across processes.
        if self.args.distributed:
            self.evaluator.synchronize_between_processes()

        miou = self.evaluator.Mean_Intersection_over_Union()
        print(f"Validation MIoU: {miou} / GPU: {self.args.rank}")
        # Only rank 0 process prints and logs the results
        if self.args.rank == 0:
            self.logger.info(f"Validation MIoU: {miou}")
            print(f"Validation MIoU: {miou}")  # Output to terminal.

            if self.earlystopping.CheckStopping(new_criterion=miou):
                # Save the best model.
                if self.args.distributed:
                    torch.save(self.model.module.state_dict(), self.best_model_path)
                else:
                    torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info("Saved the best model in epoch {}".format(epoch))
