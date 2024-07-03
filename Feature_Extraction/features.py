#import config
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from torch import nn
from tqdm import tqdm
import torch

def prepare_data_features(model, dataset, config):
    
    device = config.training.device
    if(config.post_training.transfer_learning):
        network = deepcopy(model)
    else:
        network = deepcopy(model.net)
    #network.fc = nn.Identity()
    network.eval()
    network.to(device)

    dataloader = DataLoader(dataset, batch_size=config.dataset.batch_size,
                            num_workers = config.dataset.num_workers,
                            shuffle = False,
                            drop_last = False)
    fetaures = []
    labels = []
    for images, targets in tqdm(dataloader):
        if(config.dataset.name == "TinyImagenet"):
            images = images[0].to(device)
        else:
            images = images.to(device)
        images_features = network(images)
        fetaures.append(images_features.detach().cpu())
        labels.append(targets)
    
    features = torch.cat(fetaures, dim=0)
    labels = torch.cat(labels, dim=0)

    labels, idx = labels.sort()
    features = features[idx]

    return TensorDataset(features, labels)

