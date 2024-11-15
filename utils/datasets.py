import torch
from torch.utils.data import Dataset
import os
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
        self.labels = self.cifar_data.targets

        self.classes = self.cifar_data.classes
        self.binary = len(self.classes) == 2
        
        # Calculate weights for each class
        self.class_counts = np.unique(self.labels, return_counts=True)[1]
        self.class_weights = [max(self.class_counts) / cls for cls in self.class_counts]

    def __len__(self):
        # Return the size of the dataset
        return len(self.cifar_data)

    def __getitem__(self, idx):
        # Get the data item (image and label) at the specified index
        image, label = self.cifar_data[idx]
        return image, label

