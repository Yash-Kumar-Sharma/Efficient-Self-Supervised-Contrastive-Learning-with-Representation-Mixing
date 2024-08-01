import pytorch_lightning as pl
from torchvision.datasets.fakedata import transforms
import config
from Feature_Extraction.features import prepare_data_features
from torch.utils.data import DataLoader
from data.OurModel_Data import Imagenet_Data

from Preprocess.Augmentations import model_transforms
from Preprocess.Preprocess import Preprocess
from Preprocess.DataAugmentation import DataAugmentation

class Imagenet_DataModule_le(pl.LightningDataModule):
    def __init__(self, model):
        super().__init__()

        self.K = config.K
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.dataset = config.dataset
        self.image_size = config.image_size

        self.model = model

    def prepare_data(self):     #Already Downloaded
        pass

    def setup(self, stage):
       
        transform = model_transforms(self.dataset, self.image_size)
        normalized_transform, _ = transform.GetTransform()
        #resize = torchvision.transforms.Resize(size=(self.image_size,self.image_size))
        #normalize = transforms.Normalize(mean=config.mean,
        #                                  std=config.std) #Imagenet, #STL10, #Tiny-Imagenet, #cub200-2011
        train_set = Imagenet_Data(path=config.data_dir,
                                        transform = normalized_transform,
                                        #transform = torchvision.transforms.Compose([resize,torchvision.transforms.ToTensor(),normalize]),
                                        root=config.data_dir, split='train')


        test_set = Imagenet_Data(path = config.data_dir,
                                    transform = normalized_transform,
                                    #transform = torchvision.transforms.Compose([resize,torchvision.transforms.ToTensor(),normalize]),
                                    root=config.data_dir, split='val')
        #print(len(train_set))
        #print(len(test_set))
        if(stage == 'fit'):
            self.train_data = prepare_data_features(self.model, train_set)
            print("Linear Evaluation Data Loaded ...")
        #self.val_data = prepare_data_features(self.model, val_set)
        if(stage == 'test'):
            self.test_data = prepare_data_features(self.model, test_set)
            print("Testing Data Loaded ...")
        
     
    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                           batch_size=config.batch_size,
                           num_workers=config.num_workers,
                           shuffle=True,
                           pin_memory=True)
    '''
    def val_dataloader(self):
        return DataLoader(self.val_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True)
    ''' 
    def test_dataloader(self): 
        return DataLoader(self.test_data,
                           batch_size=config.batch_size,
                           num_workers=config.num_workers,
                           shuffle=False,
                           pin_memory=True)
