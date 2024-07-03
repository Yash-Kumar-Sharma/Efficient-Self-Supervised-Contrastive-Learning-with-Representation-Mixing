import torchvision
import torch
import torch.nn as nn
import ResnetVersions

def GetBackbone(backbone_name, dataset_name):
    match backbone_name:
        case "resnet20":
            return ResnetVersions.resnet20_cifar()
        case "resnet32":
            return ResnetVersions.resnet32_cifar()
        case "resnet18":
            net = torchvision.models.resnet18(weights = None)
            net.fc = nn.Identity()
            if(dataset_name != "Imagenet"):
                net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias = False)
                net.maxpool = nn.Identity()
            return net
        case "resnet50":
            net = torchvision.models.resnet50(weights = None)
            net.fc = nn.Identity()
            if(dataset_name != "Imagenet"):
                net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias = False)
                net.maxpool = nn.Identity()
            return net

            

def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x
