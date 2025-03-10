import os
import random

import numpy as np
import torch

class BaseEngine:
    def __init__(self, args):
        """Initialize the segmentation engine.torch.cuda.set_device

        Args:
            args: Arguments containing configuration parameters.
        """
        self.args = args
        # The device is already set in main(), we just get the current device here.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        downsample_factors_str = '_'.join(map(str, args.downsample_factors))
        self.run_dir = os.path.join(
            os.getcwd(), args.run_dir, args.dataset, downsample_factors_str, args.backbone
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        self._set_seed()
        
    def _set_seed(self):
        """Sets the random seed for reproducibility."""
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

    def _build_optimizer(self):
        pass

    def _build_evaluator(self):
        pass

    def _build_loss(self):
        pass

    def _build_data(self):
        pass
    
    def train(self):
        pass

    def validate(self):
        pass

    def predict_whole(self):
        pass