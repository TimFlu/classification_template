import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1)) # Initialize temperature to 1

    def forward(self, logits):
        return logits / self.temperature
    
    def predict(self, x):
        logits = self.model(x)
        scaled_logits = self(logits)
        return scaled_logits
        

