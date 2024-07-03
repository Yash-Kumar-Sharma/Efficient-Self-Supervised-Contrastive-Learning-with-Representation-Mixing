import pytorch_lightning as pl
import torch
import os
import time
#import config
#import hydra
#from omegaconf import DictConfig

from models.OurModel import OurModel
from models.SimclrModel import SimclrModel
from models.SimclrV2Model import SimclrV2Model
from models.MocoModel import MocoModel
from models.Modified_Simclr1 import Modified_Simclr1Model
from models.Modified_Simclr2 import Modified_Simclr2Model
from models.Modified_Simclr3 import Modified_Simclr3Model
from models.SimSiamModel import SimSiamModel

#from cifar10_dataset import Cifar10_DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

from data.Imagenet.imagenet_dataset import Imagenet_DataModule
from data.TinyImagenet.tinyImagenet_dataset import TinyImagenet_DataModule
from data.Cifar100.cifar100_dataset import Cifar100_DataModule
from data.Cifar10.cifar10_dataset import Cifar10_DataModule
from data.Cifar10.cifar10_lt_dataset import Cifar10_DataModuleLT
from data.STL10.stl10_dataset import Stl10_DataModule

class PrintTime(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        batch_time = time.time() - self.start_time
        print(f"Batch time {batch_time}.")


def Get_Model(model_name):
    
    model_function = model_name + "Model"
    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']
    

def Get_Dataset(dataset_name, extension):
    dataset_function = dataset_name + "_DataModule"
    if(extension == "LT"):
        dataset_function = dataset_function + "LT"
    exec(f"generated_dataset = {dataset_function}", globals())
    return globals()['generated_dataset']
'''
class ToggleGPUCallback(pl.Callback):
    def __init__(self, multi_gpu, single_gpu):
        super().__init__()
        self.multi_gpu = multi_gpu
        self.single_gpu = single_gpu
        self.use_single_gpu = False

    def on_epoch_end(self, trainer, pl_module):
        if self.use_single_gpu:
            trainer.gpus = self.single_gpu
            print("Switching to single GPU...")
        else:
            trainer.gpus = self.multi_gpu
            print("Switching to multi-GPU...")
        self.use_single_gpu = not self.use_single_gpu
'''

#@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def Pretraining(config):
    strategy = DeepSpeedStrategy(logging_batch_size_per_gpu=config.dataset.batch_size)
    logger = TensorBoardLogger("results/pretrain_logs", name = "my_model_v1")
    profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler("results/pretrain_logs/profiler0"),
        trace_memory = True,
        schedule = torch.profiler.schedule(skip_first = 10, wait=1, warmup=1, active=20),
    )

    checkpoint_path = os.path.join("results", config.dataset.name + "_pretrain", config.feature.mode, config.imbalance.imb_type, config.backbone.name)
    pretrained_filename = os.path.join(checkpoint_path, (config.model.name + config.feature.LT + "Model.ckpt"))

    checkpoint_callback = ModelCheckpoint(dirpath=pretrained_filename,
                                          #save_weights_only=True,
                                          every_n_epochs=5,
                                          save_last=True,
                                          mode = 'min',
                                          monitor='train_loss')

    lr_monitor = LearningRateMonitor(logging_interval = 'step')

    generated_model = Get_Model(config.model.name)
    model = generated_model(config)
    generated_dataset = Get_Dataset(config.dataset.name, config.feature.LT)
    dm = generated_dataset(config)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path + config.model.name + "Model"),
        strategy = strategy,
        #profiler = profiler,
        logger = logger,
        #resume_from_checkpoint = 'cifar10/epoch=9-step=480.ckpt',
        accelerator='gpu',
        devices = config.training.devices,
        min_epochs = 1,
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],  # , print_time],
        log_every_n_steps=1,
        #reload_dataloaders_every_n_epochs=1
    )
    
    # Create ToggleGPUCallback
    #toggle_gpu_callback = ToggleGPUCallback(config.training.devices, config.training.device)
    #trainer.callbacks.append(toggle_gpu_callback)
    
    if(os.path.exists(pretrained_filename)):
        print("Model Loading..." )
        #saved_backbone = "epoch=" + str(config.training.checkpoint_toload) + "-step=" + str(config.training.checkpoint_toload) + ".ckpt"
        saved_backbone = "last.ckpt"
        #model = OurModel.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, dm, ckpt_path=os.path.join(pretrained_filename, saved_backbone))
    else:
        pl.seed_everything(42)
        trainer.fit(model, dm)
        #trainer.validate(model, dm)
    return model

