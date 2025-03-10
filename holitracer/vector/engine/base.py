import random
import numpy as np
import torch

from ..models import (
    PointRefinementModel2,
    SnakeRefineModel,
    VLRModel,
    VLRAModel,
    VLRAsModel,
)


class BaseEngine:
    def __init__(self, args):
        self.args = args
        # print args on rank 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.model = self._build_model().to(self.device)

        self._set_seed()

    def _build_model(self):
        """Builds the segmentation model.

        Raises:
            ValueError: If an invalid segmentation head is provided.
            FileNotFoundError: If the resume file is not found.

        Returns:
            The initialized model.
        """
        if self.args.rank == 0:
            print(f"Building the model with backbone: {self.args.model}")

        if self.args.model == "base":
            model = PointRefinementModel2(
                num_points=self.args.num_points,
                backbone_path=self.args.backbone_path,
                down_ratio=self.args.down_ratio,
            )
        elif self.args.model == "snake":
            model = SnakeRefineModel(
                num_points=self.args.num_points,
                backbone_path=self.args.backbone_path,
                down_ratio=self.args.down_ratio,
                snake_num=3,
            )
        elif self.args.model == "vlr":
            model = VLRModel(
                num_points=self.args.num_points,
                backbone_path=self.args.backbone_path,
                vlr_num=3,
                down_ratio=self.args.down_ratio,
            )
        elif self.args.model == "vlra":
            model = VLRAModel(
                num_points=self.args.num_points,
                backbone_path=self.args.backbone_path,
                vlr_num=4,
                down_ratio=self.args.down_ratio,
            )
        elif self.args.model == "vlras":
            model = VLRAsModel(
                num_points=self.args.num_points,
                backbone_path=self.args.backbone_path,
                vlr_num=4,
                down_ratio=self.args.down_ratio
            )
        else:
            raise ValueError(f"Invalid model: {self.args.model}")

        return model

    def _set_seed(self):
        """Sets the random seed for reproducibility."""
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

    def _build_data(self):
        pass

    def train(self):
        pass

    def validate(self, epoch):
        pass
