from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import collections
from copy import deepcopy
import os
from Pretraining.Knn_Monitor import Knn_Monitor
import utils

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, backbone, config):
        super(ModelBase, self).__init__()
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
        return x

class MocoModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()

        self.net = utils.GetBackbone(config.backbone.name, config.dataset.name)
        
        self.lr = config.training.lr
        self.epochs = config.training.max_epochs
        self.checkpoint_tosave = config.training.checkpoint_tosave
        
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

        self.knn_monitor = Knn_Monitor(config)
     
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
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
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(self.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        loss = nn.CrossEntropyLoss().to(self.device)(logits, labels)
        return loss, q, k, l_pos.mean(), l_neg.mean()

    def GetLoss(self, im1, im2):

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
        loss, pos_mean, neg_mean = self.GetLoss(train_x, train_y)

        self.loss_metric.update(loss)
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
        if(self.current_epoch % self.checkpoint_tosave == 0):
            save_path = os.path.join(self.save_path, self.model_name,
                                    "Pretrained_Model",self.dataset,
                                    self.backbone)

            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
                print("Path created...")
            file_path = os.path.join(save_path, "model" + str(self.current_epoch + 1) + ".tar")
            self.Save(file_path)
        
        top1 = self.knn_monitor.test(deepcopy(self.encoder_q.net.backbone))

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
        self.optimizer = optim.AdamW([{'params': self.parameters(), 'lr': self.lr}])
        #self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': 0.001}])

        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                      T_max = self.epochs,
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
        
