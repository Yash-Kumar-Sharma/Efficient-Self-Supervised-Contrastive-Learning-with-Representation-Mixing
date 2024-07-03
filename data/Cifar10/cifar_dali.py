from numpy.core.multiarray import array
import torch
import numpy as np

from random import shuffle

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.auto_aug import auto_augment

from base_iterator import DALIDataloader

from nvidia.dali.plugin.pytorch import DALIGenericIterator

have_cuda = torch.cuda.is_available()


class NumpySource:
    def __init__(self, *, data, targets, shuffle,
                 batch_size):
        self._data = data
        self._targets = np.array(targets)
        self._shuffle = shuffle
        self._inds = np.arange(len(self._data))
        self.batch_size = batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def targets(self):
        return self._targets

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._inds)
        self.i = 0
        self.n = len(self._data)
        return self

    def __next__(self):
        data = self._data[self.i*self.batch_size:(self.i+1)*self.batch_size]
        labels = self._targets[self.i*self.batch_size:(self.i+1)*self.batch_size]
        if not len(data):
            raise StopIteration()
        return data, labels


class ListSource:
    def __init__(self, *, data, targets, shuffle, batch_size):
        self._data = data
        self._targets = np.array(targets)
        self._shuffle = shuffle
        self._inds = np.arange(len(self._data))
        self.batch_size = batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def targets(self):
        return self._targets

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._inds)
        self.i = 0
        self.n = len(self._data)
        return self

    def reset(self):
        if self._shuffle:
            np.random.shuffle(self._inds)
        self.i = 0

    def __next__(self):
        inds = self._inds[self.i*self.batch_size:(self.i+1)*self.batch_size]
        data = self._data[inds]
        labels = self._targets[inds]
        self.i += 1
        # if len(data) < self.batch_size:
        #     self.reset()
        # if not len(data):
        #     self.i = 0
        #     self.reset()
        if len(data) < self.batch_size:
            self.reset()
            rest = self.batch_size - len(data)
            _data = np.zeros((self.batch_size, *data.shape[1:]), dtype=data.dtype)
            _data[:len(data)] = data[:]
            _data[len(data):] = self._data[self._inds[:rest]]
            _labels = np.ones(self.batch_size, dtype=labels.dtype) * -1
            _labels[:len(data)] = labels[:]
            _labels[len(data):] = self._targets[self._inds[:rest]]
            self.i = rest
            return _data.reshape(self.batch_size, -1), _labels
        return data.reshape(self.batch_size, -1), labels


@pipeline_def(enable_conditionals=True)
def cifar_train_pipeline(source, size, mean, std, dali_cpu=False):
    images, labels = fn.external_source(source=source, num_outputs=2,
                                        dtype=[types.UINT8, types.INT64])
    images = fn.reshape(images, [32, 32, 3])
    dali_device = 'cpu' if dali_cpu else 'gpu'

    images = images.gpu()
    # images = fn.paste(images,
    #                   device=dali_device,
    #                   ratio=1.25,
    #                   fill_value=0)

    images = fn.crop(images,
                     #device=dali_device,
                     crop_pos_x=fn.random.uniform(range=(0., .5)),
                     crop_pos_y=fn.random.uniform(range=(0., .5)))

    # auto_augment.auto_augment(images,
    #                           policy_name="reduced_cifar10",
    #                           shape=[size, size],
    #                           fill_value=0)

    #auto_augment.auto_augment(images,
    #                          policy_name="image_net",
    #                          shape=[size, size],
    #                          fill_value=0)


    # # color and brightness
    images = fn.hsv(images,
                    #device=dali_device,
                    hue=fn.random.uniform(range=(0, 45.)),
                    saturation=fn.random.uniform(range=(.5, 1.)))
    images = fn.brightness_contrast(images,
                                    #device=dali_device,
                                    brightness=fn.random.uniform(range=(.5, 1.)),
                                    contrast=fn.random.uniform(range=(.5, 1.)))

    # rotate
    images = fn.rotate(images.gpu(),
                       #device=dali_device,
                       angle=fn.random.uniform(range=(-10., 10.)),
                       fill_value=0)

    # # resize
    images = fn.resize(images,
                       size=size,
                       interp_type=types.INTERP_TRIANGULAR)

    # # flip was here if we didn't do mirro
    # images = fn.flip(images)

    mirror = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(images,  # .gpu() if have_cuda else images,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      mean=[x * 255 for x in mean],
                                      std=[x * 255 for x in std],
                                      mirror=mirror)
    return images, labels


@pipeline_def
def cifar_val_pipeline(source, size, mean, std, dali_cpu=False):
    images, labels = fn.external_source(source=source, num_outputs=2,
                                        dtype=[types.UINT8, types.INT64])
    images = fn.reshape(images, [32, 32, 3])
    images = fn.resize(images.gpu(),
                       size=size,
                       mode="not_smaller",
                       interp_type=types.INTERP_TRIANGULAR)
    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      mean=[x * 255 for x in mean],
                                      std=[x * 255 for x in std],
                                      mirror=False)
    return images, labels


def get_dali_loaders(cfg, train_data, val_data):
    train_source = ListSource(data=train_data.data, targets=train_data.targets,
                              shuffle=True, batch_size=cfg.batch_size)
    train_pipe = cifar_train_pipeline(train_source,
                                      size=cfg.image_size,
                                      batch_size=cfg.batch_size,
                                      mean=cfg.mean,
                                      std=cfg.std,
                                      dali_cpu=not have_cuda,
                                      num_threads=cfg.num_workers,
                                      device_id=cfg.gpu)
    
    #train_loader = DALIGenericIterator(
    #    [train_pipe],
    #    ["data", "label"],
    #    reader_name="Reader",
    #)
    
    train_loader = DALIDataloader(train_pipe,  # "TrainReader",
                                  size=len(train_data),
                                  batch_size=cfg.batch_size,
                                  output_map=["images", "labels"],
                                  # normalize=True,
                                  # mean_std=(cfg.data.mean, cfg.data.std),
                                  last_batch_policy=LastBatchPolicy.PARTIAL
                                  )
    
    val_batch_size = 128
    val_source = ListSource(data=val_data.data, targets=val_data.targets,
                            shuffle=False, batch_size=val_batch_size)
    val_pipe = cifar_val_pipeline(val_source,
                                  batch_size=val_batch_size,
                                  size=cfg.image_size,
                                  mean=cfg.mean,
                                  std=cfg.std,
                                  dali_cpu=have_cuda,
                                  num_threads=cfg.num_workers,
                                  device_id=cfg.gpu)
    
    val_loader = DALIDataloader(val_pipe,
                                size=len(val_data),
                                batch_size=val_batch_size,
                                output_map=["images", "labels"],
                                last_batch_policy=LastBatchPolicy.PARTIAL)
    

    #val_data = DALIGenericIterator(
    #    [val_pipe],
    #    ["data", "label"],
    #    reader_name="Reader",
    #)
    return {"train": train_loader, "val": val_loader}

