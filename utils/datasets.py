import torch
from torch.utils.data import Dataset
import os
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets import CIFAR10

class CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        # Load CIFAR-10 data using torchvision's CIFAR10 class
        self.cifar_data = CIFAR10(root=root, train=train, download=download, transform=transform)

        self.data = self.cifar_data.data
        self.labels = np.array(self.cifar_data.targets)

        self.classes = self.cifar_data.classes
        self.binary = len(self.classes) == 2
        
        # Calculate weights
        if self.binary:
            logger.info('datasets.py: Binary classification detected')
            self.case = 'binary'
            # Binary classification weights
            positive_counts = self.labels.sum()  # Total positive labels
            total_samples = len(self.labels)
            self.class_weights = ([
                total_samples / (2.0 * positive_counts)]
                if positive_counts > 0
                else [1.0]
            )  # Handle division by zero
        else:
            # Multiclass classification weights
            if self.labels.ndim == 1:  # Ensure labels are class indices
                logger.info('datasets.py: Multiclass classification detected')
                self.case = 'multi_class'
                class_counts = np.bincount(self.labels)
                self.class_weights = [
                    max(class_counts) / cls if cls > 0 else 1.0
                    for cls in class_counts
                ]  # Handle division by zero
            else:
                # Multi-label classification weights
                logger.info('datasets.py: Multi-label classification detected')
                self.case = 'multi_label'
                positive_counts = self.labels.sum(axis=0)  # Sum positives per class
                total_samples = self.labels.shape[0]
                self.class_weights = np.array(
                    [
                        total_samples / (2.0 * count) if count > 0 else 1.0
                        for count in positive_counts
                    ]
                )  # Handle division by zero

    def __len__(self):
        # Return the size of the dataset
        return len(self.cifar_data)

    def __getitem__(self, idx):
        # Get the data item (image and label) at the specified index
        image, label = self.cifar_data[idx]
        return image, label

