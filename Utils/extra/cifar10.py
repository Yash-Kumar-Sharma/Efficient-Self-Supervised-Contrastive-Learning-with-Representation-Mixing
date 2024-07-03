import numpy as np
from sampler import ClassAwareSampler

from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets
from Augmentations import model_transforms
from configparser import ConfigParser
import random

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,K=2,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.K = K
        #self.transform1 = transform1
        #self.transform2 = transform2
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            #import pdb
            #pdb.set_trace()
            img = self.data[selec_idx, ...]
            new_data.append(img)
            new_targets.extend([the_class, ] * the_img_num)
        
        new_data = np.vstack(new_data)
        
        self.data = new_data
        self.targets = new_targets
    '''
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
    '''
    
    #For Ours
    '''
    def __getitem__(self,index):
        img,_=self.data[index], self.targets[index]
        pic = Image.fromarray(img)
        img_list = []
        img_trans_list = []
        
        if self.transform1 is not None and self.transform2 is not None:
          for _ in range(self.K):
            img_transformed = self.transform1(pic.copy())
            img_list.append(img_transformed)

            randNumber = random.randint(0, len(self.data)-1)
            img2,_ = self.data[randNumber], self.targets[randNumber]
            pic2 = Image.fromarray(img2)

            img_transformed = self.transform1(pic2.copy())
            img_list.append(img_transformed)

            img_transformed = self.transform2(pic2.copy())
            img_trans_list.append(img_transformed)

            img_transformed = self.transform2(pic.copy())
            img_trans_list.append(img_transformed)
        else:
          raise Exception("transforms are missing...")

        return img_list, img_trans_list

    '''
    '''
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2
    '''
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class CIFAR10_LT(object):

    def __init__(self, distributed, root='./data/cifar10', imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=40):

        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        config_object = ConfigParser()
        config_object.read("con.dat")

        datainfo = config_object["data"]
         
        transform = model_transforms(datainfo["dataset"],datainfo["image_size"])
        original_train_transform, train_transform = transform.GetTransform()

        #train_dataset = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform1=train_transform,
        #        transform2 = original_train_transform)
        train_dataset = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform)

        eval_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)
        
        self.cls_num_list = train_dataset.get_cls_num_list()

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)#, drop_last = config_object.getboolean("data", "drop_last"))

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
