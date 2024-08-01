import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig

from Pretraining.Pretraining import Pretraining
from Linear_Evaluation.Transfer_Learning import Transfer_Learning
from Linear_Evaluation.Linear_Evaluation import Linear_Evaluation

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
        
    if(cfg.post_training.transfer_learning):
        Transfer_Learning(cfg)
    else:
        trained_model = Pretraining(cfg) 
        Linear_Evaluation(trained_model, cfg)

if __name__ == "__main__":
    main()
