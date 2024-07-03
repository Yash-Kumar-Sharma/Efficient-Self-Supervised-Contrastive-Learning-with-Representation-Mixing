from torch import mode
import ConfigMaster.data as data

class Training(object):
    def __init__(self, learning_rate, max_epochs, devices, checkpoint, steps):
        self._lr = learning_rate
        self._epochs = max_epochs
        self._devices = devices
        self._checkpoint = checkpoint
        self._steps = steps
    
    def _getConfig(self): 
        return {
                "lr" : self._lr, 
                "epochs" : self._epochs,
                "devices" : self._devices,
                "checkpoint" : self._checkpoint,
                "steps" : self._steps,
                }

class AugmentationConfig(object):
    def __init__(self, K, dataset):
        self._K = K
        self._mean = data.mean[dataset]
        self._std = data.std[dataset]

    def _getConfig(self):
        return{
                "K" : self._K,
                "mean" : self._mean,
                "std" : self._std,
            }

class PathConfig(object):
    def __init__(self, dataset):
        self._dataset = dataset

    def _getConfig(self):
        match self._dataset:
            case "TinyImagenet":
                data_dir = data.path["data_dir"] + "/tiny-imagenet-200/"

            case "Imagenet":
                data_dir = data.path["data_dir"] + "/imagenet/ILSVRC/"

            case _:
                data_dir = data.path["data_dir"]

        return {
            "data_dir" : data_dir,
            "data_list" : data.path["data_list"],
            "val_list" : data.path["val_list"],
            "save_path" : data.path["save_path"],
            "filepath" : data.path["filepath"],
        }

class ImbalancedConfig(object):
    def __init__(self, mode, imb_type, imb_factor):
        self._mode = mode
        self._imb_type = imb_type
        self.imb_factor = imb_factor

    def _getConfig(self):
        return {
            "mode" : self._mode,
            "imb_type" : self._imb_type,
            "imb_factor" : self._imb_factor,
        }

class DatasetConfig(AugmentationConfig, PathConfig, ImbalancedConfig):
    def __init__(self, dataset = "Cifar10", batch_size = 1024, num_workers = 32, drop_last = True):
        AugmentationConfig.__init__(self, K = 1, dataset = dataset)
        PathConfig.__init__(self, dataset= dataset)
        ImbalancedConfig.__init__(self, mode = "full", imb_type = "LT", imb_factor=0.01)

        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._drop_last = drop_last

    def _getConfig(self):
        config1  = AugmentationConfig._getConfig(self)
        config2 = PathConfig._getConfig(self)
        config3 = ImbalancedConfig._getConfig(self)

        config4 = {
                "dataset" : self._dataset,
                "image_size" : data.dataset_img_size[self._dataset],
                "batch_size" : self._batch_size,
                "drop_last" : self._drop_last,
                "num_workers" : self._num_workers,
                "num_classes" : data.dataset_categories[self._dataset]
            }

        config4.update(config3)
        config4.update(config2)
        config4.update(config1)
        return config4

class TrainingConfig(Training):
    def __init__(self, model = "Our", backbone = "resnet18"):
        super().__init__(learning_rate=0.08, max_epochs=200, devices=[0], checkpoint=10, steps=13)
        self._model = model
        self._backbone = backbone

    def _getConfig(self): 
        config1 = super()._getConfig()   
        config2 = {
                    "model_name" : self._model,
                    "backbone" : self._backbone,
                    "feature_size" : data.resnet_feature_size[self._backbone],
                    "intermediate_size" : 512,
                    "projection_size" : 128,
                    }
        config2.update(config1)
        return config2
