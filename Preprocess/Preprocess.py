import numpy as np
import torch
from kornia import image_to_tensor
from kornia.augmentation import Resize
from torch import nn, Tensor
from PIL import Image

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x = x.resize((self.image_size,self.image_size))            #For Imagenet
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0
