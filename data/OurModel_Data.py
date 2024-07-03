import torch
import torchvision
from PIL import Image
import os
import random
import numpy as np

"""
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')



def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x
"""
CIFAR-10 Dataset class - returns the pair of one image 
                          with its another random K counterpart
"""
class OurDataFromCIFAR10(torchvision.datasets.CIFAR10):
  def __init__(self, K,**kwds):
    super().__init__(**kwds)
    self.K = K              # tot number of augmentations
    #self.transform1 = transform1
    #self.transform2 = transform2

  def __getitem__(self,index):
    img,_=self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = []
    img_trans_list = []
    #if self.transform1 is not None and self.transform2 is not None:
    for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
        #img_list = torch.cat((img_list,img_transformed),1)

        randNumber = random.randint(0, len(self.data)-1)
        img2,_ = self.data[randNumber], self.targets[randNumber]
        pic2 = Image.fromarray(img2)
        
        img_transformed = self.transform(pic2.copy())
        img_list.append(img_transformed)
        #img_list = torch.cat((img_list,img_transformed),1)
        
        img_transformed = self.transform(pic2.copy())
        img_trans_list.append(img_transformed)
        #img_trans_list = torch.cat((img_trans_list,img_transformed),1)

        img_transformed = self.transform(pic.copy())
        img_trans_list.append(img_transformed)
        #img_trans_list = torch.cat((img_trans_list,img_transformed),1)
        
    #else:
      #raise Exception("transforms are missing...")
    
    data = stack(img_list,0)
    data_transform = stack(img_trans_list,0)

    
    return data, data_transform


"""
CIFAR-100 Dataset class - returns the pair of one image 
                          with its another random K counterpart
"""
class OurDataFromCIFAR100(torchvision.datasets.CIFAR100):
  def __init__(self, K,**kwds):
    super().__init__(**kwds)
    self.K = K              # tot number of augmentations
    #self.transform1 = transform1
    #self.transform2 = transform2

  def __getitem__(self,index):
    img,_=self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = []
    img_trans_list = []
    #if self.transform1 is not None and self.transform2 is not None:
    for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
        #img_list = torch.cat((img_list,img_transformed),1)

        randNumber = random.randint(0, len(self.data)-1)
        img2,_ = self.data[randNumber], self.targets[randNumber]
        pic2 = Image.fromarray(img2)
        
        img_transformed = self.transform(pic2.copy())
        img_list.append(img_transformed)
        #img_list = torch.cat((img_list,img_transformed),1)
        
        img_transformed = self.transform(pic2.copy())
        img_trans_list.append(img_transformed)
        #img_trans_list = torch.cat((img_trans_list,img_transformed),1)

        img_transformed = self.transform(pic.copy())
        img_trans_list.append(img_transformed)
        #img_trans_list = torch.cat((img_trans_list,img_transformed),1)
        
    #else:
      #raise Exception("transforms are missing...")
    
    data = stack(img_list,0)
    data_transform = stack(img_trans_list,0)

    
    return data, data_transform

"""
STL-10 Dataset class - returns the pair of one image 
                          with its another random K counterpart
"""

class OurDataFromSTL10(torchvision.datasets.STL10):
  def __init__(self, path, K,**kwds):
    super().__init__(**kwds)
    self.K = K              # tot number of augmentations
    #self.transform1 = transform1
    #self.transform2 = transform2
    #self.unlabeled = self.data
    self.unlabeled=torchvision.datasets.STL10(path,split='unlabeled')
    #print(len(self.unlabeled))

  def __getitem__(self, index):
    if(index < len(self.unlabeled)):
      pic, _ = self.unlabeled[index]
      #pic = Image.fromarray(pic)
      #pic = np.reshape(pic, (-1,96,96))

      img_list = []
      img_trans_list = []

      #if self.transform1 is not None and self.transform2 is not None:
      for _ in range(self.K):
          img_transformed = self.transform(pic.copy())
          img_list.append(img_transformed)

          randNumber = random.randint(0, len(self.unlabeled)-1)
          pic2, _  = self.unlabeled[randNumber]
          #pic2 = Image.fromarray(pic2)
          #pic = np.reshape(pic, (-1, 96,96))

          img_transformed = self.transform(pic2.copy())
          img_list.append(img_transformed)

          img_transformed = self.transform(pic2.copy())
          img_trans_list.append(img_transformed)

          img_transformed = self.transform(pic.copy())
          img_trans_list.append(img_transformed)
      #else:
      #  raise Exception("transforms are missing...")

      #return img_list, img_trans_list
      
    data = stack(img_list,0)
    data_transform = stack(img_trans_list,0)

    
    return data, data_transform



"""
TinyImagenet Dataset class - returns the pair of one image
                          with its another random K counterpart
"""

class OurDataFromTinyImagenet(torch.utils.data.Dataset):
  def __init__(self, K, root, data_list, transform, loader = default_loader):
    super().__init__()

    images = []
    labels = open(data_list).readlines()
    items_label=0
    for line in labels:
      items = line.strip('\n').split()
      img_folder_name = os.path.join(root, "train/",items[0],"images/")

      for filename in os.listdir(img_folder_name):
        each = os.path.join(img_folder_name, filename)
        if(os.path.isfile(each)):
                images.append((each,items_label))
        else:
            print(each + 'Not Found')
      items_label = items_label + 1

      self.K = K
      self.root = root
      self.images = images
      self.transform = transform
      #self.transform1 = transform1
      #self.transform2 = transform2
      self.loader = loader

  def __getitem__(self, index):
    img_name,_=self.images[index]
    img = self.loader(img_name)
    img_list = []
    img_trans_list = []
    #if self.transform1 is not None:
    img_transformed = self.transform(img.copy())
    img_list.append(img_transformed)
    for _ in range(self.K):
        randNumber = random.randint(0, len(self.images)-1)
        img2_name,_ = self.images[randNumber]
        img2 = self.loader(img2_name)
        img_transformed = self.transform(img2.copy())
        img_list.append(img_transformed)
        img_transformed = self.transform(img2.copy())
        img_trans_list.append(img_transformed)

    img_transformed = self.transform(img.copy())
    img_trans_list.append(img_transformed)

    #return img_list, img_trans_list

    data = stack(img_list,0)
    data_transform = stack(img_trans_list,0)

    
    return data, data_transform
  
  def __len__(self):
        return len(self.images)

class OurDataFromImagenet(torchvision.datasets.ImageNet):
    def __init__(self, K, path, split, **kwds):
        super().__init__(**kwds)
        # import pdb
        # pdb.set_trace()
        self.K = K  # tot number of augmentations
        self.path = path
        self.split = split
        self.data = torchvision.datasets.ImageNet(root = self.path, split=self.split)
        #self.transform = transform
        #self.transform2 = transform2
        #self.device = device

    def __getitem__(self, index):
        #for index in range(int(len(self.data)/1000)):
        pic, target = self.data[index]
        # randNumber = random.randint(0, 50000-1)
        # img2, target2 = self.data[randNumber], self.targets[randNumber]
        # pic = Image.fromarray(img)
        # pic2 = Image.fromarray(img2)
        # print(index)
        img_list = []
        img_trans_list = []
        #if self.transform is not None:
        for _ in range(self.K):
            img_transformed = self.transform(pic.copy())
            img_list.append(img_transformed)

            randNumber = random.randint(0, len(self.data)-1)
            pic2, target2 = self.data[randNumber]

            img_transformed = self.transform(pic2.copy())
            img_list.append(img_transformed)

            img_transformed = self.transform(pic2.copy())
            img_trans_list.append(img_transformed)

            img_transformed = self.transform(pic.copy())
            img_trans_list.append(img_transformed)
        #else:
        #    raise Exception("transforms are missing...")

        data = stack(img_list,0)
        data_transform = stack(img_trans_list,0)

        
        return data, data_transform


class Imagenet_Data(torchvision.datasets.ImageNet):
    def __init__(self, path, **kwds):
        super().__init__(**kwds)
        # import pdb
        # pdb.set_trace()
        self.path = path
        #self.split = split
        self.data = torchvision.datasets.ImageNet(root = self.path)

    def __getitem__(self, index):
        #for index in range(int(len(self.data)/1000)):
        pic, target = self.data[index]
        #print(pic)
        # randNumber = random.randint(0, 50000-1)
        # img2, target2 = self.data[randNumber], self.targets[randNumber]
        # pic = Image.fromarray(pic)
        # pic2 = Image.fromarray(img2)
        # print(index)
        #img_list = []
        #img_trans_list = []
        if self.transform is not None:
            pic = self.transform(pic)
            #print(pic)
        else:
            raise Exception("transforms are missing...")

        return pic, target

