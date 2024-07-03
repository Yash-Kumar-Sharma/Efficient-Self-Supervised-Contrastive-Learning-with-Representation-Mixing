from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from Loss.loss import xent_loss
from Utils.metric.similarity_mean import Positive_Negative_Mean
#import config
import torchmetrics
import torch.nn.functional as F
from functools import reduce
import operator
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import collections
from copy import deepcopy
import os

def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, backbone, config):
        super(ModelBase, self).__init__()
        #self.traininfo, self.datainfo, self.mocoinfo = GetMocoInfo()
        #self.info = info
        #self.net = utils.GetResnetNetwork(self.datainfo["arch"])
        self.net = nn.Sequential(collections.OrderedDict([
          ("backbone", backbone)
        ]))
        self.net.fc = nn.Linear(config.backbone.feature_size,config.model.moco_dim)

        self.head = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(config.model.moco_dim, config.model.moco_dim*4)),
          ("bn1",      nn.BatchNorm1d(config.model.moco_dim*4)),
          ("relu1",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(config.model.moco_dim*4, config.model.projection_size)),

        ]))
        
        
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        # note: not normalized here
        return x

class MocoModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False

        self.net = torchvision.models.resnet18(weights = None)
        self.net.fc = nn.Identity()
        #self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias = False)
        #self.net.maxpool = nn.Identity()
        
        self.lr = config.training.lr
        self.max_epochs = config.dataset.max_epochs
        self.batch_size = config.dataset.batch_size
        self.image_size = config.dataset.image_size
        self.filepath = config.dataset.filepath
        self.save_path = config.dataset.save_path
        self.dataset = config.dataset.name
        
        self.backbone = config.backbone.name
        self.dim = config.backbone.feature_size
        
        self.model_name = config.model.name
        self.K = config.model.moco_K
        self.m = config.model.moco_m
        self.T = config.model.moco_T
        self.symmetric = config.model.moco_Symmetric
        
        # create the encoders
        self.encoder_q = ModelBase(self.net, config)
        
        self.encoder_k = ModelBase(deepcopy(self.net), config)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(config.model.projection_size, self.K))
        #self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        
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

     
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(self.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        loss = nn.CrossEntropyLoss().to(self.device)(logits, labels)
        return loss, q, k, l_pos.mean(), l_neg.mean()

    def GetLoss(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2, p_mean, l_mean = self.contrastive_loss(im1, im2)
            loss_21, q2, k1, p_mean, l_mean = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
            #print(k.shape)
        else:  # asymmetric loss
            loss, q, k, p_mean, l_mean = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss, p_mean, l_mean

    def training_step(self, batch, batch_idx): 
         
        train_x, train_y = batch
        self.optimizer.zero_grad()
        
        #self.embeddings = self.net(train_x)
        
        loss, pos_mean, neg_mean = self.GetLoss(train_x, train_y)

        self.loss_metric.update(loss)
        #pos_mean, neg_mean = Positive_Negative_Mean(x = self.embeddings, device = self.global_rank, batch_size = self.batch_size)
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
         
            y2 = train_y[:4]
            grid_y = torchvision.utils.make_grid(y2.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("Cifar_B", grid_y, self.current_epoch)
        

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
        
        encoder_q_state_dict = self.encoder_q.state_dict()
        encoder_k_state_dict = self.encoder_k.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save({"encoder_q": encoder_q_state_dict,
                    "encoder_k": encoder_k_state_dict,
                    "optimizer": optimizer_state_dict},
                    model_path)
        print("model saved succesfully ..")

    """
    Function to load Our Model
    """

    def Load(self, file_path, LinearEvaluation = False):
        checkpoint = torch.load(file_path,map_location='cpu')
        if(LinearEvaluation):
            self.encoder_q.net.load_state_dict(checkpoint["encoder_q"]["backbone"])
        else:
            self.encoder_q.load_state_dict(checkpoint["encoder_q"])
            self.encoder_k.load_state_dict(checkpoint["encoder_k"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
