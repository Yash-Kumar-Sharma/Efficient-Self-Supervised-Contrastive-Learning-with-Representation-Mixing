from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from Loss.Modified1_loss import xent_loss
from metric.similarity_mean import Modified_Positive_Negative_Mean
import torchmetrics
import torch.nn.functional as F
from functools import reduce
import operator
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x

class Modified_Simclr2Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False
        
        self.net = torchvision.models.resnet18(weights = None, num_classes = config.model.projection_size * 4)
        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias = False)
        self.net.maxpool = nn.Identity()
        
        self.net.fc = nn.Sequential(
            self.net.fc,
            nn.BatchNorm1d(config.model.projection_size * 4),
            nn.LeakyReLU(),
            nn.Linear(config.model.projection_size * 4, config.model.projection_size),
        )

        self.lr = config.training.lr
        self.max_epochs = config.training.max_epochs
        
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

    def forward(self, x):
        x = self.net(x)
        return x
   
    def training_step(self, batch, batch_idx): 
        
        train_x, train_x_transform = batch
        self.optimizer.zero_grad()
        embeddings = self.net(train_x)
        embeddings_transform = self.net(train_x_transform)

        newEmbeddings = torch.add(embeddings,embeddings_transform)
        loss = xent_loss(newEmbeddings, embeddings)

        self.loss_metric.update(loss)
        pos_mean, neg_mean = Modified_Positive_Negative_Mean(x = newEmbeddings, y = embeddings, device = self.global_rank)
        self.pos_mean += pos_mean * len(train_x)
        self.neg_mean += neg_mean * len(train_x_transform)
        self.total += len(train_x) + len(train_x_transform)

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
        if(self.current_epoch % 10 == 0):
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

        if(self.current_epoch + 1 == config.max_epochs):
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
            
            #self.mean_weights_per_epoch.clear()
        
    ''' 
    def validation_step(self, batch, batch_idx): 
        train_x, _ = batch
        
        #data = torch.stack(x,dim=1)
        #data_transform = torch.stack(y,dim=1)
        #d = data.size()

        #train_x = data.view(d[0]*2*self.K, d[2],d[3],d[4])
        #train_x_transform = data_transform.view(d[0]*2*self.K, d[2],d[3],d[4])
        
        embeddings = self.forward(train_x)
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        #embeddings_transform = self.forward(train_x_transform)
        #norm_embeddings_transform = torch.nn.functional.normalize(embeddings_transform, p=2, dim=1)

        #newEmbeddings = torch.add(norm_embeddings, norm_embeddings_transform)
        #norm_embeddings = torch.nn.functional.normalize(newEmbeddings, p=2, dim=1)
        loss = xent_loss(norm_embeddings)

        self.log_dict(
            {
                'val_loss': loss,
                'step': self.current_epoch,
            },
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            sync_dist=True,
        )
    '''
    def configure_optimizers(self):
        self.optimizer = optim.AdamW([{'params': self.parameters(), 'lr': self.lr}])
        #self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': 0.001}])

        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                      T_max = config.max_epochs,
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

        
        
