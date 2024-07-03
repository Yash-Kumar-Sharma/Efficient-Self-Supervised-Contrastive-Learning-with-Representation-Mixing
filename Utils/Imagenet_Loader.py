import torchvision
import random
from nvidia.dali.pipeline import Pipeline
#import nvidia.dali.ops as ops
#import nvidia.dali.types as types
#import nvidia.dali.fn as fn

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os

class MixingImagenet(torchvision.datasets.ImageNet):
    def __init__(self, data, transform1, transform2, K,
                 root="", split="", device=None):
        super().__init__(root=root, split=split)
        # import pdb
        # pdb.set_trace()
        self.K = K              # tot number of augmentations
        self.data = data
        self.transform1 = transform1
        self.transform2 = transform2
        self.device = device

    def __getitem__(self, index):
        #for index in range(int(len(self.data)/1000)):
        pic, target = self.data[index]
        # randNumber = random.randint(0, 50000-1)
        # img2, target2 = self.data[randNumber], self.targets[randNumber]
        # pic = Image.fromarray(img)
        # pic2 = Image.fromarray(img2)
        # print(index)
        img_list = []
        img_trans_list = []
        if self.transform1 is not None and self.transform2 is not None:
            #for _ in range(self.K):
                img_transformed = self.transform1(pic.copy())
                img_list.append(img_transformed)

                randNumber = random.randint(0, len(self.data)-1)
                pic2, target2 = self.data[randNumber]

                img_transformed = self.transform1(pic2.copy())
                img_list.append(img_transformed)

                img_transformed = self.transform2(pic2.copy())
                img_trans_list.append(img_transformed)

                img_transformed = self.transform2(pic.copy())
                img_trans_list.append(img_transformed)
        else:
            raise Exception("transforms are missing...")

        return img_list, img_trans_list

'''
class get_dali_pipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(get_dali_pipeline, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = fn.readers.file(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = fn.decoders.image_random_crop(device="mixed", output_type=types.RGB)
        self.res = fn.resize(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = fn.crop_mirror_normalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            #image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        #self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        #images = self.res(images)
        #output = self.cmnp(images, mirror=rng)
        
        img_list = []
        img_trans_list = []
        #if self.transform1 is not None and self.transform2 is not None:
        #for _ in range(self.K):
        #img_transformed = self.transform1(pic.copy())
        img_transformed = self.res(images)
        img_list.append(img_transformed)

        randNumber = random.randint(0, 1281167-1)
        jpegs2, target2 = self.input(name = "Reader")
        pic2 = self.decode(jpegs2)    
        #img_transformed = self.transform1(pic2.copy())
        img_transformed = self.res(pic2)
        img_list.append(img_transformed)

        #img_transformed = self.transform2(pic2.copy())
        img_transformed = self.res(pic2)
        img_trans_list.append(img_transformed)

        #img_transformed = self.transform2(pic.copy())
        img_transformed = self.res(images)
        img_trans_list.append(img_transformed)
        #else:
        #    raise Exception("transforms are missing...")

        return img_list, img_trans_list


        #return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

'''
@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline():
    images_dir = "/home/gen/yash/OurData/data/imagenet/ILSVRC/train/" 
    img_list = list()
    img_transformed_list = list()

    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, name="Reader")
    copy_image = images
    # decode data on the GPU
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())
     
    #img_list.append(images)

    #images_transformed, labels = fn.readers.file(
    #    file_root=images_dir, random_shuffle=True, name="Reader")
    # decode data on the GPU
    images_transformed = fn.decoders.image_random_crop(
        copy_image, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images_transformed = fn.resize(images_transformed, resize_x=224, resize_y=224)
    images_transformed = fn.crop_mirror_normalize(
        images_transformed,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())

    #img_transformed_list.append(images_transformed)
    
    images_T, labels_T = fn.readers.file(
        file_root=images_dir, random_shuffle=True)
    images_T_copy = images_T
    # decode data on the GPU
    images_T = fn.decoders.image_random_crop(
        images_T, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images_T = fn.resize(images_T, resize_x=224, resize_y=224)
    images_T = fn.crop_mirror_normalize(
        images_T,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())
    
    #img_list.append(images_T)
    
    #images_transformed, labels = fn.readers.file(
    #    file_root=images_dir, random_shuffle=True, name="Reader")
    # decode data on the GPU
    images_T_transformed = fn.decoders.image_random_crop(
        images_T_copy, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images_T_transformed = fn.resize(images_T_transformed, resize_x=224, resize_y=224)
    images_T_transformed = fn.crop_mirror_normalize(
        images_T_transformed,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())
    
    #img_transformed_list.append(images_T_transformed)
    
    return images, images_T_transformed, images_T, images_transformed
    #return images, images_transformed, images_T,0

















































