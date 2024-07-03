import pytorch_lightning as pl  
import torchvision
from data.OurModel_Data import OurDataFromCIFAR100
from data.Simclr_Data import SimclrDataFromCIFAR100
from data.Moco_Data import MocoDataFromCIFAR100

#import config
from torch.utils.data import DataLoader
from Preprocess.Preprocess import Preprocess
from Preprocess.DataAugmentation import DataAugmentation

class Cifar100_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
         
        self.model_name = config.model.name

        self.K = config.dataset.K
        self.data_dir = config.dataset.data_dir
        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers
        self.drop_last = config.dataset.drop_last
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size
        self.crop_max = config.dataset.crop_max
        self.mean = config.dataset.mean
        self.std = config.dataset.std


        self.preprocess = Preprocess(image_size = self.image_size)
        
        filter = int(0.1 * self.image_size)
        if(filter % 2 == 0):
            kernel_size = filter - 1
        else:
            kernel_size = filter
        
        self.transform1 = DataAugmentation(image_size = self.image_size,
                                           kernel_size = kernel_size,
                                           crop_max = self.crop_max,
                                           mean = self.mean,
                                           std = self.std,
                                           apply_normalize_only = True)

        self.transform2 = DataAugmentation(image_size = self.image_size,
                                           kernel_size = kernel_size,
                                           crop_max = self.crop_max,
                                           mean = self.mean,
                                           std = self.std,
                                           apply_normalize_only = False)
    def prepare_data(self):     #Already Downloaded
        '''       
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, '.')
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')
        '''

    def setup(self, stage):
        if(self.model_name == "Our"):
            print("Our Data")
            self.train_set = OurDataFromCIFAR100(K=self.K,
                                     root = self.data_dir,
                                     transform = self.preprocess,
                                     train = True,
                                     download = False)
        
        if(self.model_name == "Simclr" or self.model_name =="SimclrV2" or
               self.model_name == "Modified_Simclr1" or self.model_name == "Modified_Simclr2" or
               self.model_name == "Modified_Simclr3"):
            print("Simclr Data")
            self.train_set = SimclrDataFromCIFAR100(K=self.K,
                                     root = self.data_dir,
                                     transform = self.preprocess,
                                     train = True,
                                     download = False)
        
        if(self.model_name == "Moco"):
            print("Moco Data")
            self.train_set = MocoDataFromCIFAR100(root = self.data_dir,
                                     transform = self.preprocess,
                                     train = True,
                                     download = False,
                                     )
        
        #self.train_set, self.val_set = random_split(entire_set, [40000, 10000])
        #self.train_set = ExternalCIFAR10PreTrainIterator(data=entire_set, batch_size=config.batch_size)
 
    def on_after_batch_transfer(self, batch, dataloader_idx):

        if(self.model_name == "Our"):     
            data, data_transform = batch        
            d = data.size()
            train_x = data.view(d[0]*2*self.K, d[2],d[3], d[4])
            train_x_transform = data_transform.view(d[0]*2*self.K, d[2],d[3], d[4])
            train_x = self.transform1(train_x)
            train_x_transform = self.transform2(train_x_transform)
            return train_x,train_x_transform
        
        if(self.model_name == "Simclr" or self.model_name == "SimclrV2" or
               self.model_name == "Modified_Simclr1" or self.model_name == "Modified_Simclr2" or
               self.model_name == "Modified_Simclr3"):
            data, target = batch        
            d = data.size()
            train_x = data.view(d[0]*self.K, d[2],d[3], d[4])
            #train_x_transform = data_transform.view(d[0]*2*self.K, d[2],d[3], d[4])
            train_x = self.transform2(train_x)
            #train_x_transform = self.transform2(train_x_transform)
            
            return train_x,target
        
        if(self.model_name == "Moco"):
            data_1, data_2 = batch        
            #d = data.size()
            #train_x = data.view(d[0]*self.K, d[2],d[3], d[4])
            #train_x_transform = data_transform.view(d[0]*2*self.K, d[2],d[3], d[4])
            train_x = self.transform2(data_1)
            train_y = self.transform2(data_2)
            #train_x_transform = self.transform2(train_x_transform)
            
            return train_x,train_y
        
    def train_dataloader(self):
        #return self.dataloaders['train']
        
        return DataLoader(self.train_set,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          drop_last=self.drop_last,
                          pin_memory = False,
                          )
        
        
    '''
    
    def val_dataloader(self):
        #return self.dataloaders['val']
        
        return DataLoader(self.val_set,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True,
                      drop_last=False)
    '''    
