import torch
import torchvision
from PIL import Image
import os


def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x

class ModifiedDataFromCIFAR10(torchvision.datasets.CIFAR10):
  def __init__(self, K, **kwds):
    super().__init__(**kwds)
    self.K = K # tot number of augmentations

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    normalizedImg = self.transform(pic.copy())

    img_list = list()
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
    else:
        img_list = img
    data = stack(img_list,0)
    
    return data, normalizedImg


