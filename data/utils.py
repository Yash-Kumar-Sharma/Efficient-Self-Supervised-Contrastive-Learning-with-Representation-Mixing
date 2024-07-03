
from Preprocess.DataAugmentation import DataAugmentation

def GetAugmentations(image_size, crop_max, mean, std):
    
    filter = int(0.1 * image_size)
    if(filter % 2 == 0):
        kernel_size = filter - 1
    else:
        kernel_size = filter
    
    normalized_transform = DataAugmentation(image_size = image_size,
                                       kernel_size = kernel_size,
                                       crop_max = crop_max,
                                       mean = mean,
                                       std = std,
                                       apply_normalize_only = True)

    augmented_transform = DataAugmentation(image_size = image_size,
                                       kernel_size = kernel_size,
                                       crop_max = crop_max,
                                       mean = mean,
                                       std = std,
                                       apply_normalize_only = False)

    return normalized_transform, augmented_transform

def ApplyTransforms(batch, model_name, K, transform1, transform2):
     
    match model_name:
        case "Our":
            data, data_transform = batch        
            d = data.size()
            train_x = data.view(d[0]*2*K, d[2],d[3], d[4])
            train_x_transform = data_transform.view(d[0]*2*K, d[2],d[3], d[4])
            train_x = transform1(train_x)
            train_x_transform = transform2(train_x_transform)
            return train_x,train_x_transform
        
        case "Simclr" | "SimclrV2":
            data, target = batch        
            d = data.size()
            train_x = data.view(d[0]*K, d[2],d[3], d[4])
            train_x = transform2(train_x)
            return train_x,target
        
        case "Moco":
            data_1, data_2 = batch        
            train_x = transform2(data_1)
            train_y = transform2(data_2)
            return train_x,train_y

        case _:
            raise Exception("Available Models are - Our, Simclr, SimclrV2, Moco")
