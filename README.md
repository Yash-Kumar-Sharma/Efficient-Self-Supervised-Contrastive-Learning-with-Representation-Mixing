# Efficient-Self-Supervised-Contrastive-Learning-with-Representation-Mixing
   This is an official implementations of the paper "Efficient Self Supervised Contrastive Learning with Representation Mixing".

## Requirements

   Python              - 3.10.12 <br />
   Tensorboard         - 2.13.0  <br />
   Pytorch             - 1.12.0+cu116 <br />
   Pytorch-lightning   - 2.3.3 <br />

## config settings

 config

    config.yaml

    backbone
        resnet18.yaml
        resnet20.yaml
        resnet32.yaml
        resnet50.yaml

    dataset
        Cifar100.yaml
        Cifar10.yaml
        Imagenet.yaml
        Stl10.yaml
        TinyImagenet.yaml

    model
        Moco.yaml
        Modified1.yaml
        Modified2.yaml
        Modified3.yaml
        Our.yaml
        Simclr.yaml

    post_training
        linear_evaluation.yaml
        transfer_learning.yaml

    training
        defauly.yaml
        pretraining.yaml

## How to Run

### (Pretraining & linear_evaluation)
    
    <p>python3 main.py dataset.data_dir="path_to_dataset" dataset.save_path="path_to_save_model_on_each_nth_epoch"</p> <br />
    <p>Note - Default dataset is Cifar10, Default model is Simclr, Default backbone is resnet20 </p><br />

####or 

   As per the config files hierarchy use command line arguments to feed new values

####or

   Make changes in respective config file and then run - python3 main.py

### (Transfer_Learning)

   Run python3.main post_training=transfer_learning <br />
   Note - Modify post_training/transfer_learning as per the requirements <br />
   It works like - source dataset = transfer_learning.transfer_from, target_dataset = dataset (from config.yaml) <br />

## You can access the tensorboard logs using

   tensorboard --logdir results/pretrain_logs/

