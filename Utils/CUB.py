from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np

def read_image(path):
    im = cv2.imread(str(path))
    return Image.fromarray(im)
    #im = cv2.imread(str(path))
    #return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def prepare_CUB(path):
    PATH = Path(path)
    labels = pd.read_csv(PATH/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]

    train_test = pd.read_csv(PATH/"train_test_split.txt", header=None, sep=" ")
    train_test.columns = ["id", "is_train"]

    images = pd.read_csv(PATH/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    classes = pd.read_csv(PATH/"classes.txt", header=None, sep=" ")
    classes.columns = ["id", "class"]

    return labels, train_test, images, classes

class CUB(Dataset):
    def __init__(self, files_path, labels, train_test, image_name, train=True, 
                 transform=False):
      
        self.files_path = files_path
        self.labels = labels
        self.transform = transform
        self.train_test = train_test
        self.image_name = image_name
        
        if train:
          mask = self.train_test.is_train.values == 1
          
        else:
          mask = self.train_test.is_train.values == 0
        
        
        self.filenames = self.image_name.iloc[mask]
        self.labels = self.labels[mask]
        self.num_files = self.labels.shape[0]
       
      
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        y = self.labels.iloc[index,1] - 1
        file_name = self.filenames.iloc[index, 1]
        path = os.path.join(self.files_path,'images',file_name)
        
        x = read_image(path)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = cv2.resize(x, (224,224))
        #x = normalize(x)
        #x =  np.rollaxis(x, 2) # To meet torch's input specification(c*H*W) 
        return x,y
