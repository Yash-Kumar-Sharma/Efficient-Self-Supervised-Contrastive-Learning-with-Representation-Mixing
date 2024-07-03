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
"""
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')


class SimclrV2DataFromCIFAR10(torchvision.datasets.CIFAR10):
  def __init__(self, K, **kwds):
    super().__init__(**kwds)
    self.K = K # tot number of augmentations

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = list()
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
    else:
        img_list = img
    data = stack(img_list,0)
    return data, target

class MultiCIFAR100(torchvision.datasets.CIFAR100):
  def __init__(self, K, **kwds):
    super().__init__(**kwds)
    self.K = K # tot number of augmentations

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = list()
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
    else:
        img_list = img
    return img_list, target

class MultiSTL10(torchvision.datasets.STL10):
    def __init__(self , K, path, **kwds):
        super().__init__(**kwds)
        # import pdb
        # pdb.set_trace()
        self.K = K              # tot number of augmentations
        #self.labeled=datasets.STL10("./data",split='train',download=True)
        self.unlabeled=torchvision.datasets.STL10(path,split='unlabeled')

    def __getitem__(self, index):
        if(index < len(self.unlabeled)):
            pic,_=self.unlabeled[index]
            img_list = []

            if self.transform is not None:
                for _ in range(self.K):
                    img_transformed = self.transform(pic.copy())
                    img_list.append(img_transformed)
            else:
                img_list = pic

        return img_list, False

class MultiTinyImagenet(torch.utils.data.Dataset):
  def __init__(self, K, root, data_list, transform, loader = default_loader,**kwds):
    #super().__init__(**kwds)
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
      self.loader = loader

  def __getitem__(self, index):
    img_name,_=self.images[index]
    img = self.loader(img_name)
    img_list = []
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(img.copy())
        img_list.append(img_transformed)

    return img_list, False

  def __len__(self):
    return len(self.images)


class DataFromIMAGENET(torchvision.datasets.ImageNet):
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
        img_list = list()
        img_trans_list = []
        #if self.transform is not None:
        for _ in range(self.K):
            img_transformed = self.transform(pic.copy())
            img_list.append(img_transformed)
        #else:
        #    raise Exception("transforms are missing...")

        data = stack(img_list,0) 
        return data, target


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

