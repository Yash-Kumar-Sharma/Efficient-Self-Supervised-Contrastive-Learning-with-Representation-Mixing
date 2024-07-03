Cifar10_mean = [0.491, 0.482, 0.447]
Cifar10_std = [0.247, 0.243, 0.262]

Cifar100_mean = [0.5071, 0.4867, 0.4408] 
Cifar100_std = [0.2675, 0.2565, 0.2761]

Imagenet_mean = [0.485,0.456,0.406]
Imagenet_std = [0.229,0.224,0.225]

STL10_mean = [0.485,0.456,0.406]
STL10_std = [0.485,0.456,0.406]

TinyImagenet_mean = [0.485,0.456,0.406]
TinyImagenet_std = [0.229,0.224,0.225]

Cub_mean = [0.229,0.224,0.225]
Cub_std = [0.229,0.224,0.225]

path = {
    "data_dir" : "/home/gen/yash/OurData/data/",
    "data_list" : "/home/gen/yash/OurData/data/tiny-imagenet-200/wnids.txt",
    "val_list" : "/home/gen/yash/OurData/data/tiny-imagenet-200/val/out.txt",

    "save_path" : "/home/gen/yash/work/Result_zip/LT-Results",
    "filepath" : "results/MeanWeights",
}

resnet_feature_size = {
    "resnet20" : 64,
    "resnet32" : 64,
    "resnet18" : 512,
    "resnet50" : 2048,
}

dataset_img_size = {
    "Cifar10" : 32,
    "Cifar100" : 32,
    "Stl10" : 96,
    "TinyImagenet" : 64,
    "Imagenet" : 224,
}

dataset_categories = {
    "Cifar10" : 10,
    "Cifar100" : 100,
    "Stl10" : 10,
    "TinyImagenet" : 200,
    "Imagenet" : 1000,
}

mean = {
    "Cifar10" : Cifar10_mean,
    "Cifar100" : Cifar100_mean,
    "Stl10" : STL10_mean,
    "TinyImagenet" : TinyImagenet_mean,
    "Imagenet" : Imagenet_mean,
}

std = {
    "Cifar10" : Cifar10_std,
    "Cifar100" : Cifar100_std,
    "Stl10" : STL10_std,
    "TinyImagenet" : TinyImagenet_std,
    "Imagenet" : Imagenet_std,
}
