from torch.utils.data import random_split
import torchvision
from dali_cifar10 import ExternalCIFAR10iterator, ExternalSourcePipeline, ExternalCIFAR10PreTrainIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy, DALIGenericIterator
import config
from torch.utils.data import DataLoader
from base_iterator import DALIDataloader

class CIFAR10_DataModule():
    def __init__(self):
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def GetDataLoader(self):
        entire_set = torchvision.datasets.CIFAR10(root = self.data_dir,
                                          #transform = torchvision.transforms.ToTensor(),
                                          train = True,
                                          download = False)
        
        train_set, val_set = random_split(entire_set, [40000, 10000])


        dali_train_set = ExternalCIFAR10PreTrainIterator(data = train_set, batch_size = int(self.batch_size), transform = torchvision.transforms.ToTensor()) 
        
        pipe_train = ExternalSourcePipeline(batch_size = self.batch_size,
                                                external_data = dali_train_set,
                                                num_threads = 8,
                                                device_id = 0,
                                                size = config.image_size,
                                                std = config.std,
                                                mean = config.mean
                                                )
        pipe_train.build()
        train_loader = DALIDataloader(pipe_train,  # "TrainReader",
                                  size=len(train_set),
                                  batch_size=self.batch_size,
                                  output_map=["images", "aug_images", "images2", "aug_images2"],
                                  # normalize=True,
                                  # mean_std=(cfg.data.mean, cfg.data.std),
                                  last_batch_policy=LastBatchPolicy.DROP
                                  )
        return train_loader
