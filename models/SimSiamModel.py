from tensorboardX import torchvis
from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR 
from Loss.SimSiam_Loss import SimSiamLoss
import math
#from Utils.metric.similarity_mean import SimSiam_Positive_Negative_Mean
#import config
import torchmetrics
import torch.nn.functional as F
from functools import reduce
import operator
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from resnet_cifar import ResNet18
from Pretraining.Knn_Monitor import Knn_Monitor
from copy import deepcopy

def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class SimSiamModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.net = torchvision.models.resnet18(weights = None, zero_init_residual=True)
        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias = False)
        #self.net.relu = nn.Identity() 
        self.net.maxpool = nn.Identity()
        
        self.net.fc = nn.Sequential(
            nn.Linear(config.backbone.feature_size, config.model.projection_size, bias = False),
            nn.BatchNorm1d(config.model.projection_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.model.projection_size, config.model.projection_size, bias = False),
            nn.BatchNorm1d(config.model.projection_size)
        )
        self.predictor = prediction_MLP(in_dim=config.model.projection_size)
        self.criterian = SimSiamLoss()
        
        self.lr = config.training.lr
        self.max_epochs = config.training.max_epochs
        self.weight_decay = config.training.weight_decay
        self.steps = config.training.steps
        self.warmup_lr = config.training.warmup_lr
        self.warmup_epochs = config.training.warmup_epochs
        self.base_lr = config.training.base_lr
        self.final_lr = config.training.final_lr
        self.single_gpu = config.training.device
        self.multi_gpu = config.training.devices
        
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
        self.layer_names = []
        self.mean_weights_per_epoch = {name: [] for name, module in self.net.named_modules() if isinstance(module, torch.nn.Conv2d)}
        self.conv_layer_configs = []
        self.loss_value = []

        self.knn_monitor = Knn_Monitor(config)

        self.use_single_gpu = False


    def forward(self, x):
        x = self.net(x)
        return x
   
    def training_step(self, batch, batch_idx): 
        train_x1, train_x2 = batch
        #import pdb
        #pdb.set_trace()
        self.optimizer.zero_grad()
        z1 = self.net(train_x1)
        z2 = self.net(train_x2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss1 = -nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss2 = -nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean()

        loss = loss1/2 + loss2/2

        self.loss_metric.update(loss)
        
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
            y1 = train_x1[:4]
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
        '''
        if(self.use_single_gpu == False):
            print("single")
            self.trainer.gpus = self.single_gpu
            self.use_single_gpu = True
        '''
        #else:
        #   self.training.gpus = self.multi_gpu
        #self.use_single_gpu = not self.use_single_gpu
        #save model .. 
        if((self.current_epoch+1) % 10 == 0):
            save_path = os.path.join(self.save_path, self.model_name,
                                    "Pretrained_Model",self.dataset,
                                    self.backbone,"pytorch_lightning")

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
        '''
        if(self.use_single_gpu == True):
            print("multi")
            self.trainer.gpus = self.multi_gpu
            self.use_single_gpu = False
        '''
    def configure_optimizers(self):
        #self.optimizer = optim.AdamW([{'params': self.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}])
        self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay}])

        #self.scheduler = CosineAnnealingLR(self.optimizer, 
        #                              T_max = self.max_epochs,
        #                              eta_min=0, last_epoch=-1)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda i: 0.5 * (math.cos(i * math.pi / self.max_epochs) + 1))
        '''
        self.scheduler = LR_Scheduler(
                                    self.optimizer,
                                    self.warmup_epochs, self.warmup_lr*self.batch_size/256, 
                                    self.max_epochs, self.base_lr*self.batch_size/256, self.final_lr*self.batch_size/256, 
                                    self.trainer.num_training_batches,
                                    constant_predictor_lr=True # see the end of section 4.2 predictor
                                )
        '''
        #return self.optimizer
        
        #self.scheduler = OneCycleLR(self.optimizer, max_lr = self.lr, epochs = self.max_epochs, steps_per_epoch = self.steps)
        
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

        
        
