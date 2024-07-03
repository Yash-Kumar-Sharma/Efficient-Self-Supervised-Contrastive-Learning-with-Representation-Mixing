#Dataset Processing
from Simclr.data.Simclr_Data import *
#from Simclr.Simclr_Data import MultiCIFAR100
#from Utils.Simclr_Data import MultiSTL10
#from Utils.Simclr_Data import MultiTinyImagenet

from Moco.data.Moco_Data import *
#from Utils.Moco_Data import CIFAR100Pair
#from Utils.Moco_Data import STL10Pair
#from Utils.Moco_Data import TinyImagenetPair

from Ours.data.OurModel_Data import *
##from Utils.OurModel_Data import MixingDataFromCIFAR100
#from Utils.OurModel_Data import MixingDataFromSTL10
#from Utils.OurModel_Data import MixingDataFromTinyImagenet
#from Utils.OurModel_Data import MixingDataFromCUB
from Modified1.data.Modified_Data import *

import torchvision
import os
import Utils.CUB_Loader as CUB_Loader
import Utils.Imagenet_Loader as Imagenet_Loader

def GetCIFAR10Train_Set(model_name, K, path, train_transform, original_train_transform):
  train_set = ''
  match model_name:
      case "Ours" | "Modified2" | "Modified3" :
        train_set = MixingDataFromCIFAR10(K=K,root=path,train=True,
                      transform1=train_transform,
                      transform2=original_train_transform,
                      #path="./data",
                      download =True,
                      )
      case "Simclr" | "SDCLR":
        train_set = MultiCIFAR10(K=K,root=path,train=True,
                      transform=train_transform,
                      #path="./data",
                      download =True,
                      )
      case "Moco" | "BYOL":
        train_set = CIFAR10Pair(root=path,train=True,
                      transform=train_transform,
                      #path="./data",
                      download =True,
                      )
        #print("train_set ")
      case "Modified":
        train_set = NewCIFAR10(K=K,root=path,train=True,
                      transform1=train_transform,
                      transform2 = original_train_transform,
                      #path="./data",
                      download =True,
                      )
      
      case "Modified2":
        train_set = MixingDataFromCIFAR10(K=K,root=path,train=True,
                      transform1=train_transform,
                      transform2=original_train_transform,
                      #path="./data",
                      download =True,
                      )
      case "Modified3":
        train_set = MixingDataFromCIFAR10(K=K,root=path,train=True,
                      transform1=train_transform,
                      transform2=original_train_transform,
                      #path="./data",
                      download =True,
                      )

  return train_set

def GetCIFAR100Train_Set(model_name, K, path, train_transform, original_train_transform):
  train_set = ''
  match model_name:
      case "Ours":
        train_set = MixingDataFromCIFAR100(K=K,root=path,train=True,
                      transform1=train_transform,
                      transform2=original_train_transform,
                      #path="./data",
                      download =True,
                      )
      case "Simclr" | "SDCLR":
        train_set = MultiCIFAR100(K=K,root=path,train=True,
                      transform=train_transform,
                      #path="./data",
                      download =True,
                      )
      case "Moco":
        train_set = CIFAR100Pair(root=path,train=True,
                      transform=train_transform,
                      #path="./data",
                      download =True,
                      )

  return train_set

def GetSTL10Train_Set(model_name, K, path, train_transform, original_train_transform):
  train_set = ''
  match model_name:
      case "Ours":
        train_set = MixingDataFromSTL10(K=K,root=path,split = 'unlabeled',
                      transform1=train_transform,
                      transform2=original_train_transform,
                      path=path,
                      download =True,
                      )
      case "Simclr" | "SDCLR":
        train_set = MultiSTL10(K=K,root=path,split = 'unlabeled',
                      transform=train_transform,
                      path=path,
                      download =True,
                      )
      case "Moco":
        train_set = STL10Pair(root=path,split = 'unlabeled',
                      transform=train_transform,
                      path=path,
                      download =True,
                      )

  return train_set

def GetTinyImagenetTrain_Set(model_name, K, path, data_list, train_transform, original_train_transform):
  train_set = ''
  match model_name:
      case "Ours":
        train_set = MixingDataFromTinyImagenet(K=K,root=path,
                      data_list = data_list,
                      transform1=train_transform,
                      transform2=original_train_transform,
                      )

      case "Simclr" | "SDCLR":
        train_set = MultiTinyImagenet(K=K,root=path, data_list = data_list,
                      transform=train_transform,
                      download =True,
                      )
      case "Moco":
        train_set = TinyImagenetPair(root=path, data_list = data_list,
                      transform=train_transform,
                      download =True,
                      )

  return train_set

def GetImagenetTrain_Set(model_name, K, path, train_transform, original_train_transform, datadir):
    train_set = ''
    match model_name:
        case "Ours":
            train_dataset = torchvision.datasets.ImageFolder(os.path.join(datadir,"train"))
            #import pdb
            #pdb.set_trace()
            train_set = Imagenet_Loader.MixingImagenet(data=train_dataset,
                                       transform1=train_transform,
                                       transform2=original_train_transform,
                                       K=K, root=datadir, split="train")
            #import pdb
            #pdb.set_trace()
    return train_set


def GetCUBTrain_Set(model_name, K, path, train_transform, original_train_transform, datadir):
    train_set = ''
    match model_name:
        case "Simclr":
            print("taken")
            train_set = CUB_Loader.MultiCUB(root = datadir, K=K, transform=train_transform,
                                       target_transform=original_train_transform,
                                       train = True)
        case "Ours":
            #train_dataset = torchvision.datasets.ImageFolder(os.path.join(datadir))
            #import pdb
            #pdb.set_trace()
            train_set = CUB_Loader.MixingCUB(root = datadir, K=K, transform=train_transform,
                                       target_transform=original_train_transform,
                                       train = True)
            #import pdb
            #pdb.set_trace()
    return train_set


