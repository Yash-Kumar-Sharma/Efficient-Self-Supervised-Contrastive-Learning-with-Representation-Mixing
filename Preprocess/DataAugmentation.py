import torch
from omegaconf import OmegaConf
from torch import nn
from torch import Tensor
from kornia.augmentation import ColorJiggle, ColorJitter, RandomHorizontalFlip, RandomResizedCrop, Normalize, RandomGrayscale
from kornia.filters.gaussian import GaussianBlur2d

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, image_size, kernel_size, crop_max, mean, std, apply_normalize_only):
        super().__init__()
        self.mean = OmegaConf.to_object(mean)
        self.std = OmegaConf.to_object(std)
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.crop_max = crop_max
        self.apply_normalize_only = apply_normalize_only
        self.normalize = Normalize(mean = self.mean, std = self.std)
        
        self.transforms = nn.Sequential(
            #ColorJiggle(brightness=0.8, contrast=0.8,saturation=0.8, hue=0.2),
            ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            RandomGrayscale(p=0.2),
            #RandomResizedCrop(size=self.image_size, scale=(0.08, self.crop_max)),
            RandomResizedCrop(size=(self.image_size, self.image_size),
                              scale=(0.08, self.crop_max)),
            RandomHorizontalFlip(p=0.5),
            
            GaussianBlur2d(kernel_size = (self.kernel_size, self.kernel_size), sigma = (0.1, 2.0)),
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        if(not self.apply_normalize_only):
            x = self.transforms(x)  # BxCxHxW
        x_out = self.normalize(x)
        
        return x_out
