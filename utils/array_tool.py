import numpy as np
import torch

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()