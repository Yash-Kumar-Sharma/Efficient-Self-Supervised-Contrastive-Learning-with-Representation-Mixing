#Augmentations
crop_max = 1

#Dataset
dataset = "Cifar10"
extension = ""  
data_dir = "/home/gen/yash/OurData/data/"
data_list = "/home/gen/yash/OurData/data/tiny-imagenet-200/wnids.txt"
val_list = "/home/gen/yash/OurData/data/tiny-imagenet-200/val/out.txt"

save_path = "/home/gen/yash/work/Result_zip/LT-Results"
drop_last = False

filepath = "results/MeanWeights"
image_size = 32
K = 1 
batch_size = 1024
num_workers = 32
min = 0
max = 1

#Imbalanced
imb_factor = 0.01
imb_type = "balanced"

#Selection
mode = "partial"

#Model Pretraining
model_name = "Our"
backbone = "resnet18"
feature_size = 512
intermediate_size = 512
projection_size = 128
lr = 0.08
device = "cuda:0"
max_epochs = 1
devices = [0,1,2,3]
checkpoint = 4
steps = 13

#Common In Pretraining and linear_evaluation
weight_decay = 1e-4

#Moco
moco_K = 4096
moco_m = 0.99
moco_T = 0.1
moco_Symmetric = False
moco_dim = 128

#Model linear_evaluation
linear_evaluation_lr = 0.008
linear_evaluation_max_epochs = 1
checkpoint_ll = 99
num_classes = 10
steps_le = 49

#Normalization
Cifar10_mean = [0.491, 0.482, 0.447]
Cifar10_std = [0.247, 0.243, 0.262]

Cifar100_mean = [0.5071, 0.4867, 0.4408] 
Cifar100_std = [0.2675, 0.2565, 0.2761]

Imagenet_std = [0.485,0.456,0.406]
Imagenet_mean = [0.229,0.224,0.225]

STL10_std = [0.485,0.456,0.406]
STL10_std = [0.485,0.456,0.406]

TinyImagenet_std = [0.485,0.456,0.406]
TinyImagenet_mean = [0.229,0.224,0.225]

Cub_mean = [0.229,0.224,0.225]
Cub_mean = [0.229,0.224,0.225]

