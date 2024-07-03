import pytorch_lightning as pl
import torch
import config
from OurModel import OurModel
from Cifar10.cifar10_dataset import Cifar10_DataModule
import torchvision
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def train_cifar10(config_import, data_dir, num_epochs, num_gpus):
#if __name__ == "__main__":
    #device = torch.device('cuda')
    torch.set_float32_matmul_precision('medium')

    strategy = DeepSpeedStrategy(logging_batch_size_per_gpu=config.batch_size)
    logger = TensorBoardLogger("pretrain_logs", name = "my_model_v1")
    profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler("pretrain_logs/profiler0"),
        trace_memory = True,
        schedule = torch.profiler.schedule(skip_first = 10, wait=1, warmup=1, active=20),
    )
    #backbone = torchvision.models.resnet18(weights = None)
    #backbone.fc = nn.Identity()

    model = OurModel(config_import)
    '''
    dm = Cifar10_DataModule(K = config.K,
                            data_dir = data_dir,
                            dataset = config.dataset,
                            image_size = config.image_size,
                            batch_size = config.batch_size,
                            num_workers = config.num_workers)
    '''
    dm = Cifar10_DataModule()
    metrics = {"train_loss": "train_loss", "p_mean": "pos_mean", "n_mean": "neg_mean"}
    trainer = pl.Trainer(
        strategy = strategy,
        profiler = profiler,
        logger = logger,
        accelerator='gpu',
        devices = num_gpus,
        min_epochs = 1,
        max_epochs=num_epochs,
        precision = "32-true",
        callbacks=[TuneReportCallback(metrics, on = "train_end")],
        log_every_n_steps=1,
    )

    trainer.fit(model, dm)
