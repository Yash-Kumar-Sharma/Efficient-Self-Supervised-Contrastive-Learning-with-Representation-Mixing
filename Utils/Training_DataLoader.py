from configparser import ConfigParser
from Utils.Support.Augmentations import model_transforms
import utils
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch

def GetTrainLoader():

    config_object = ConfigParser()
    config_object.read("Config/con.dat")

    datainfo = config_object["data"]
    traininfo = config_object["train"]
    auginfo = config_object["augmentations"]
    linearinfo = config_object["linear"]

    transform = model_transforms(datainfo["dataset"],datainfo["image_size"])
    original_train_transform, train_transform = transform.GetTransform()
    train_set = utils.GetTrainSet(dataset = datainfo["dataset"],
                                model_name = datainfo["model_name"],
                                K = int(auginfo["K"]),
                                path = datainfo["path"],
                                train_transform = train_transform,
                                original_train_transform = original_train_transform,
                                data_list = datainfo["data_list"],
                                datadir = datainfo["datadir"])

    train_length = len(train_set)
    indices=list(range(len(train_set)))
    loading_data = 1 - float(datainfo["dataset_images"])/100
    split = int(np.floor(loading_data * train_length))

    np.random.shuffle(indices)

    train_idx=indices[split:]

    train_sampler=SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(train_set,
                                      pin_memory = True,
                                      batch_size=int(traininfo["batch_size"]),
                                      num_workers=int(traininfo["num_workers"]),
                                      #shuffle=True,
                                      sampler = train_sampler,
                                      #sampler = DistributedSampler(train_set),
                                      drop_last = config_object.getboolean("data", "drop_last"))

    return train_loader

