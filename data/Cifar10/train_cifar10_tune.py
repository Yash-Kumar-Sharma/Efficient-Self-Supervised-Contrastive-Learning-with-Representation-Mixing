#import tempfile
from ray import tune
from Train_OurModel import Train_OurModel

num_samples = 10
num_epochs = 10
gpus_per_trial = 1 # set this to higher if using GPU

#data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
# Download data
#MNISTDataModule(data_dir=data_dir).prepare_data()

config = {
    "intermediate_size": tune.choice([512]),
    "projection_size": tune.choice([128]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([1024]),
}

trainable = tune.with_parameters(
    Train_OurModel,
    data_dir="/home/gen/yash/OurData/data/",
    num_epochs=num_epochs,
    num_gpus=gpus_per_trial)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 32,
        "gpu": gpus_per_trial
    },
    metric="train_loss",
    mode="min",
    config=config,
    num_samples=num_samples,
    name="tune_cifar10")

print(analysis.best_config)
