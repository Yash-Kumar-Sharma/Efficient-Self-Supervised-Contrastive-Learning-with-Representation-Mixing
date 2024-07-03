import torch
import torchvision
from PIL import Image
import os

"""
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')


class MocoDataFromCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2
class MocoDataFromCIFAR100(torchvision.datasets.CIFAR100):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2
class MocoDataFromSTL10(torchvision.datasets.STL10):
    """CIFAR10 Dataset.
    """
    def __init__(self, path, **kwds):
        super().__init__(**kwds)
        self.unlabeled=torchvision.datasets.STL10(path,split='unlabeled')

    def __getitem__(self, index):
        img,_ = self.unlabeled[index]
        #img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2
class MocoDataFromTinyImagenet(torch.utils.data.Dataset):
  def __init__(self, root, data_list, transform, loader = default_loader,**kwds):
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

      #self.K = K
      self.root = root
      self.images = images
      self.transform = transform
      self.loader = loader

  def __getitem__(self, index):
    img_name,_=self.images[index]
    img = self.loader(img_name)
    #img_list = []
    if self.transform is not None:
      #for _ in range(self.K):
        #img_transformed = self.transform(img.copy())
        #img_list.append(img_transformed)
        img_1 = self.transform(img)
        img_2 = self.transform(img)

    return img_1, img_2

  def __len__(self):
    return len(self.images)

class MocoDataFromImagenet(torchvision.datasets.ImageNet):
    def __init__(self, path, split, **kwds):
        super().__init__(**kwds)
        # import pdb
        # pdb.set_trace()
        self.path = path
        self.split = split
        self.data = torchvision.datasets.ImageNet(root = self.path, split=self.split)
        #self.transform = transform
        #self.transform2 = transform2
        #self.device = device

    def __getitem__(self, index):
        pic, _ = self.data[index]
        img_1 = self.transform(pic)
        img_2 = self.transform(pic)
        return img_1, img_2
