from torch import nn, optim
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
#import config
import torchmetrics
import os

from sklearn.metrics import precision_score , recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class linearlayer_training(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.model_name = config.model.name
        self.backbone = config.backbone.name
        self.linear_layer = nn.Linear(config.backbone.feature_size, config.dataset.num_classes)
        
        self.dataset = config.dataset.name
        self.save_path = config.dataset.save_path
        self.batch_size = config.dataset.batch_size
        self.num_classes = config.dataset.num_classes

        self.lr = config.post_training.lr
        self.epochs = config.post_training.max_epochs
        self.steps = config.post_training.steps
        self.weight_decay = config.post_training.weight_decay
        
        self.mode = config.feature.mode
        self.imb_type = config.imbalance.imb_type

        
        self.outputs = []
        self.val_outputs = []
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.accuracy_top5 = torchmetrics.Accuracy(task = 'multiclass', num_classes = self.num_classes, top_k=5)
        self.loss_metric = torchmetrics.MeanMetric()
        self.criterion = nn.CrossEntropyLoss()

        self.y_test = []
        self.pred = []
        self.train_accuracy = []
        self.test_accuracy = []

    def forward(self, x):
        x = self.linear_layer(x)
        return x

    def on_train_epoch_start(self):
        self.y_test.clear()
        self.pred.clear()

    def training_step(self, batch, batch_idx):
        #self.optimizer.zero_grad()

        data, target = batch

        output = self.forward(data)
        loss = self.criterion(output, target)
        self.loss_metric.update(loss)
        #self.manual_backward(loss)
        #self.optimizer.step()
        #self.scheduler.step()
        _, predicted = torch.max(output.data, 1)
        self.y_test.extend(target.cpu())
        self.pred.extend(predicted.cpu())

        self.log_dict(
            {
                'linear_evaluation_loss': self.loss_metric.compute(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': self.current_epoch,
            },
            on_step = True,
            on_epoch = False,
            prog_bar = True,
        )
        
        self.outputs.append({'output': output, 'label': target})        
        
        return loss
   
    def on_train_epoch_end(self):
        output = torch.cat([x["output"] for x in self.outputs])
        target = torch.cat([x["label"] for x in self.outputs])
        accuracy = self.accuracy(output, target)
        #accuracy_top5 = self.accuracy_top5(output, target)

        self.log_dict(
            {
                'Linear_Evaluation_Acc': accuracy,
                #'Acc_Top5': accuracy_top5,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.loss_metric.reset()
        self.accuracy.reset()
        #self.accuracy_top5.reset()
        self.outputs.clear()
   
    def on_train_end(self):
        cm = multilabel_confusion_matrix(self.y_test, self.pred)
        numerator = [x[0][0] + x[1][1] for x in cm]
        denominator = [np.sum(x) for x in cm]
        for i in range(10):
            self.train_accuracy.append(numerator[i] / denominator[i])
        self.y_test.clear()
        self.pred.clear()
        #self.Save_LinearLayer()
    '''
    def validation_step(self, batch, batch_idx):    
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        accuracy = self.accuracy(output, target)

        self.log_dict(
            {
                'val_loss': loss,
                'val_accuracy': accuracy,
                'step': self.current_epoch,
            },
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            sync_dist=True,
        )
    '''
    def test_step(self, batch, batch_idx): 
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        accuracy = self.accuracy(output, target)
        accuracy_top5 = self.accuracy_top5(output, target)
        
        _, predicted = torch.max(output.data, 1)
        self.y_test.extend(target.cpu())
        self.pred.extend(predicted.cpu())
        
        self.log_dict(
            {
                'test_loss': loss,
                'test_accuracy': accuracy,
                'test_accuracy_top5': accuracy_top5,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            #sync_dist=True,
        )
        

    def on_test_epoch_end(self):
        #import pdb
        #pdb.set_trace()
        #self.y_test = [item.cpu().tolist() for item in self.y_test]
        #self.pred = [item.cpu().tolist() for item in self.pred]

        acc_per_class = torchmetrics.functional.accuracy(torch.Tensor(self.y_test), torch.Tensor(self.pred), task='multiclass',num_classes=self.num_classes,average = 'none')
        print(acc_per_class)
        cm = multilabel_confusion_matrix(self.y_test, self.pred)
        print(cm)
        numerator = [x[0][0] + x[1][1] for x in cm]
        denominator = [np.sum(x) for x in cm]
        for i in range(10):
            self.test_accuracy.append(numerator[i] / denominator[i])
        
        acc_out = ["%.2f" % x for x in self.test_accuracy]
        acc_train_out = ["%.2f" % x for x in self.train_accuracy]

        print(acc_train_out)
        print(acc_out)
        sklearn_precision = precision_score(self.y_test,self.pred,average='weighted')
        print("Precision = ",sklearn_precision)

        sklearn_recall = recall_score(self.y_test,self.pred, average='weighted')
        print("Recall = ",sklearn_recall)

        sklearn_f1_score = f1_score(self.y_test,self.pred, average='weighted')
        print("F1_score = ",sklearn_f1_score)
    
    def Save_LinearLayer(self):

        #save linear layer .. 
        save_path = os.path.join(self.save_path, self.model_name,
                                "Pretrained_LinearLayer",self.dataset,
                                self.backbone,self.mode, self.imb_type)

        if(not os.path.exists(save_path)):
            os.makedirs(save_path)
            print("Path created...")
        file_path = os.path.join(save_path, "linear.tar")
        
        self.Save(file_path)
        
    
    """
    Function to save Our Model
    """
     
    def Save(self, linear_layer_path):
        
        linear_layer_state_dict = self.linear_layer.state_dict()

        torch.save({"linear_layer": linear_layer_state_dict},
                    linear_layer_path)
        print("Lienar layer saved succesfully ..")
    
    def configure_optimizers(self):
        self.optimizer = optim.AdamW([{'params': self.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}])
        #self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay}])

        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                      T_max = self.epochs,
                                      eta_min=0, last_epoch=-1)
        #return [self.optimizer], [self.scheduler]
        
        #self.scheduler = OneCycleLR(self.optimizer, max_lr = self.lr, epochs = self.epochs, steps_per_epoch = self.steps)
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'Linear_Evaluation_Acc',
                'interval': 'step',
                'frequency': 1,
            }
        }
        
        
