import pytorch_lightning as pl
import torch
import os
import time
import config
from SimclrV2Model import SimclrV2Model
#from cifar10_dataset import Cifar10_DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
#from Imagenet.imagenet_dataset import Imagenet_DataModule
#from TinyImagenet.tinyImagenet_dataset import TinyImagenet_DataModule
#from Cifar100.cifar100_dataset import Cifar100_DataModule
from Cifar10.cifar10_dataset import Cifar10_DataModule
#from STL10.stl10_dataset import Stl10_DataModule

class PrintTime(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        batch_time = time.time() - self.start_time
        print(f"Batch time {batch_time}.")


#from ray.tune.integration.pytorch_lightning import TuneReportCallback
def Train_SimclrV2Model():
    strategy = DeepSpeedStrategy(logging_batch_size_per_gpu=config.batch_size)
    logger = TensorBoardLogger("pretrain_logs", name = "my_model_v1")
    profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler("pretrain_logs/profiler0"),
        trace_memory = True,
        schedule = torch.profiler.schedule(skip_first = 10, wait=1, warmup=1, active=20),
    )
    checkpoint_path = "cifar10_pretrain"
    pretrained_filename = os.path.join(checkpoint_path, "SimclrV2Model.ckpt")

    checkpoint_callback = ModelCheckpoint(dirpath=pretrained_filename,
                                          #save_weights_only=True,
                                          every_n_epochs=5,
                                          save_last=True,
                                          #mode = 'min',
                                          #monitor='train_loss'
                                          )

    lr_monitor = LearningRateMonitor(logging_interval = 'step')

    #checkpoint_path = "cifar10_pretrain"
    model = SimclrV2Model()

    #dm = Cifar100_DataModule()
    dm = Cifar10_DataModule()
    #dm = Stl10_DataModule()
    #dm = Imagenet_DataModule()
    #dm = TinyImagenet_DataModule()
    # print_time = PrintTime()

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path + "SimclrV2Model"),
        strategy = strategy,
        #profiler = profiler,
        logger = logger,
        #resume_from_checkpoint = 'cifar10/epoch=9-step=480.ckpt',
        accelerator='gpu',
        devices = [0,1,2,3],
        min_epochs = 1,
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],  # , print_time],
        log_every_n_steps=1,
        #reload_dataloaders_every_n_epochs=1
    )
    #pretrained_filename = os.path.join(checkpoint_path, "OurModel.ckpt")

    if(os.path.exists(pretrained_filename)):
        print("SimclrV2Model Loading..." )
        #model = OurModel.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, dm, ckpt_path=os.path.join(pretrained_filename, "epoch=199-step=199.ckpt"))
    else:
        pl.seed_everything(42)
        trainer.fit(model, dm)
        #trainer.validate(model, dm)
    return model

