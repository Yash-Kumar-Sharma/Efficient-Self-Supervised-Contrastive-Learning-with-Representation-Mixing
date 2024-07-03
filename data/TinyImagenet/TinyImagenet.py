import torch
import os
from PIL import Image
"""
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')


class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, data_list, train=True, transform = None, loader = default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        super().__init__()
        self.transform = transform
        images = []
        labels = open(data_list).readlines()
        #import pdb
        #pdb.set_trace()
        items_label=0
        if(train):
            for line in labels:
                items = line.strip('\n').split()
                img_folder_name = os.path.join(root, "train/",items[0],"images/")

                #print(img_name)
                # test list contains only image name
                #test_flag = True if len(items) == 1 else False
                #label = None if test_flag == True else np.array(int(items[1]))

                for filename in os.listdir(img_folder_name):
                    each = os.path.join(img_folder_name, filename)
                    if(os.path.isfile(each)):
                        images.append((each,items_label))
                    else:
                        print(each + 'Not Found')
                items_label = items_label + 1
                #if os.path.isfile(os.path.join(root, "train/",img_name)):
                #    images.append((img_name, label))
                #else:
                #    print(os.path.join(root,'train/' ,img_name) + 'Not Found.')
        else:
            img_folder_name = os.path.join(root, "val/images/")
            for line in labels:
                items = line.strip('\n').split()
                #import pdb
                #pdb.set_trace()
                #for filename in os.listdir(img_folder_name):
                each = os.path.join(img_folder_name,items[0])
                if(os.path.isfile(each)):
                    images.append((each,int(items[1])))
                else:
                    print(each + 'Not Found')
        #import pdb
        #pdb.set_trace()
        #self.K = K
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        #img = Image.fromarray(img_name).convert('RGB')
        #import pdb
        #pdb.set_trace()
        img = self.loader(img_name)
        #img = Image.fromarray(img)
        raw_img = img.copy()
        img_list = list()
        if self.transform is not None:
            #for _ in range(self.K):
            img_trans = self.transform(raw_img)
            img_list.append(img_trans)

        return (img_list, label) if label is not None else img_list

    def __len__(self):
        return len(self.images)


