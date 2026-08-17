import random
import torch
import torch_npu
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch_npu.npu.is_available():
        torch_npu.npu.manual_seed_all(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)