import pytorch_lightning as pl
#import config
from Linear_Evaluation.Model_LE import linearlayer_training
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import torch
from data.Cifar100.cifar100_dataset_le import Cifar100_DataModule_le
from data.Cifar10.cifar10_dataset_le import Cifar10_DataModule_le
from data.STL10.stl10_dataset_le import Stl10_DataModule_le
from data.Imagenet.imagenet_dataset_le import Imagenet_DataModule_le
from data.TinyImagenet.tinyImagenet_dataset_le import TinyImagenet_DataModule_le
import utils

def Get_Model(model_name):
   
    model_function = model_name + "Model_LE"

    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']
    

def Get_Dataset(dataset_name):
    dataset_function = dataset_name + "_DataModule_le"
    exec(f"generated_dataset = {dataset_function}", globals())
    return globals()['generated_dataset']

def Transfer_Learning(config):
     
    linear_checkpoint_path = os.path.join("results", config.dataset.name + "_linear")
    logger = TensorBoardLogger("results/pretrain_logs", name = "my_model_v1")
    
    pretrained_filename = os.path.join(linear_checkpoint_path, (config.model.name + "ModelLE.ckpt"))
    
    checkpoint_callback = ModelCheckpoint(dirpath=pretrained_filename,
                                                 #save_weights_only=True,
                                                 mode = "max",
                                                 monitor='Linear_Evaluation_Acc')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = linearlayer_training(config)
    generated_dataset = Get_Dataset(config.dataset.name)
    trained_model = utils.GetBackbone(config.backbone.name, config.dataset.name)
    print(config.post_training.transfer_from + " -----> " + config.dataset.name)
    file_to_load = model + config.post_training.checkpoint_toload + ".tar" 
    save_path = os.path.join(config.dataset.save_path, config.model.name,
                            "Pretrained_Model",config.post_training.transfer_from,
                            config.backbone.name, file_to_load)
    print(save_path)
    checkpoint = torch.load(save_path)
    trained_model.load_state_dict(checkpoint["model"])
    dm = generated_dataset(trained_model, config)
    

    trainer = pl.Trainer(
        default_root_dir=os.path.join(linear_checkpoint_path, (config.model.name + "ModelLE")),
        logger = logger,
        accelerator='gpu',
        devices = config.post_training.devices,
        min_epochs = 1,
        max_epochs=config.post_training.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
    )
    
    pl.seed_everything(42)
    trainer.fit(model, dm)
    trainer.test(model, dm)
