from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from Loss.loss import xent_loss
from metric.similarity_mean import Positive_Negative_Mean
#import config
import torchmetrics
import torch.nn.functional as F
from functools import reduce
import operator
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import utils
from Pretraining.Knn_Monitor import Knn_Monitor
from copy import deepcopy

class SimclrModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.net = utils.GetBackbone(config.backbone.name, config.dataset.name)
        
        self.head = nn.Sequential(
            nn.Linear(config.backbone.feature_size, config.model.projection_size * 4),
            nn.BatchNorm1d(config.model.projection_size * 4),
            nn.LeakyReLU(),
            nn.Linear(config.model.projection_size * 4, config.model.projection_size),
        )
        self.lr = config.training.lr
        self.max_epochs = config.training.max_epochs
        self.weight_decay = config.training.weight_decay

        self.K = config.dataset.K
        self.batch_size = config.dataset.batch_size
        self.image_size = config.dataset.image_size
        self.filepath = config.dataset.filepath
        self.save_path = config.dataset.save_path
        self.dataset = config.dataset.name

        self.model_name = config.model.name
        self.backbone = config.backbone.name
        
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = config.dataset.num_classes)
        self.loss_metric = torchmetrics.MeanMetric()
        self.outputs = []
        self.val_outputs = []
        self.pos_mean = 0.0
        self.neg_mean = 0.0
        self.total = 0
        # Initialize lists to store layer names and mean weights per epoch
        self.knn_monitor = Knn_Monitor(config)
        self.layer_names = []
        self.mean_weights_per_epoch = {name: [] for name, module in self.net.named_modules() if isinstance(module, torch.nn.Conv2d)}
        self.conv_layer_configs = []
        self.loss_value = []

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x
   
    def training_step(self, batch, batch_idx): 
        
        train_x,_ = batch
        self.optimizer.zero_grad()
        self.embeddings = self.forward(train_x)
        
        loss = xent_loss(self.embeddings)

        self.loss_metric.update(loss)
        pos_mean, neg_mean = Positive_Negative_Mean(x = self.embeddings, device = self.global_rank)
        self.pos_mean += pos_mean * len(train_x)
        self.neg_mean += neg_mean * len(train_x)
        self.total += len(train_x)

        self.log_dict(
            {
                'train_loss': self.loss_metric.compute(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': self.current_epoch,

            },
            on_step = True,
            on_epoch = False,
            prog_bar = True,
            sync_dist=True,
        )
        
        if(batch_idx % 100 == 0):
            y1 = train_x[:4]
            grid_x = torchvision.utils.make_grid(y1.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("Cifar_A", grid_x, self.current_epoch)
        
        return loss
    # Function to count convolutional layers
    def count_conv_layers(self, model):
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                count += 1
        return count

    # Function to record convolutional layer configurations
    def record_conv_configs(self, model):
        configs = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                configs.append((name, module.kernel_size, module.stride, module.padding))
        return configs    

    def on_train_epoch_end(self):
        #pos_mean, neg_mean = Positive_Negative_Mean(x = self.embeddings, device = self.global_rank, batch_size = self.batch_size)
         
        self.log_dict(
            {
                'pos_mean': self.pos_mean/self.total,
                'neg_mean': self.neg_mean/self.total,
                #'MeanWeight': sum(weights) / len(weights),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.loss_value.append(self.loss_metric.compute().cpu().numpy())
        self.loss_metric.reset()
        self.accuracy.reset()
        #self.loss_meter = torch.tensor(0)
        self.pos_mean = 0.0
        self.neg_mean = 0.0
        self.total = 0
        
        #save model .. 
        if((self.current_epoch + 1) % 10 == 0):
            save_path = os.path.join(self.save_path, self.model_name,
                                    "Pretrained_Model",self.dataset,
                                    self.backbone)

            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
                print("Path created...")
            file_path = os.path.join(save_path, "model" + str(self.current_epoch + 1) + ".tar")
            self.Save(file_path)
        
        configs = self.record_conv_configs(self.net)
        self.conv_layer_configs.append(configs)

        if(self.current_epoch + 1 == self.max_epochs):
            convfilename = os.path.join(self.filepath, ("file.pkl"))
            lossfilename = os.path.join(self.filepath, "loss.txt")
            
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)

            filename = os.path.join(convfilename)
            file = open(filename,"wb")
            lossfile = open(lossfilename, "wb")
            pickle.dump(self.conv_layer_configs, file)
            pickle.dump(self.loss_value, lossfile)
            lossfile.close()
            file.close()
            print("Saved")
            
        
        top1 = self.knn_monitor.test(deepcopy(self.net))

        self.log_dict(
            {
                'Knn Top-1': top1,
                #'Knn Top-5': top5,
            },
            on_epoch = True,
            prog_bar = True,
            sync_dist=True,
        )
            #self.mean_weights_per_epoch.clear()
        
    def configure_optimizers(self):
        self.optimizer = optim.AdamW([{'params': self.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}])
        #self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': 0.001}])

        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                      T_max = self.max_epochs,
                                      eta_min=0, last_epoch=-1)
        #return self.optimizer
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'train_loss',
                'interval': 'step',
                'frequency': 1,
            }
        }
        
    
    """
    Function to save Our Model
    """
     
    def Save(self, model_path):
        
        model_state_dict = self.net.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save({"model": model_state_dict,
                    "optimizer": optimizer_state_dict},
                    model_path)
        print("model saved succesfully ..")

        
        
