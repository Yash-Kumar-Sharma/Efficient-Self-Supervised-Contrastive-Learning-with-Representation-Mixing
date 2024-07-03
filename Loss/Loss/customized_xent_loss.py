import torch
import math
import numpy as np
import config

def xent_loss(x, t=0.5, eps=1e-8):
    # Estimate cosine similarity
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    x = (x @ x.t()) / (n * n.t()).clamp(min=eps)

    result = torch.where((x > config.min) & (x < config.max))
    x_new = x[result]
    #x_new = x_new.tolist()
    #import pdb
    #pdb.set_trace()
    wide_n = int(math.sqrt(x_new.shape[0]))

    #y = torch.ones(wide_n, wide_n)
    #x_new = random.sample(x_new, wide_n * wide_n)
    x_new = x_new[:wide_n*wide_n]
    #x_new = np.array(x_new)
    #x = torch.from_numpy(x_new)
    x = x_new.reshape(wide_n, wide_n)

    '''
    dim_2 = result[1]
    #import pdb
    #pdb.set_trace()
    dim_2 = dim_2.tolist()
    dim2 = random.sample(dim_2, int(int(self.info["batch_size"])/int(self.info["div"])))
    j = 0
    for each_dim in dim2:
        y[j] = x[each_dim][dim2]
        j = j + 1
    
    #import pdb
    #pdb.set_trace()
    x = y
    
    test = x[x>float(self.info["data_range_min"])]
    test = test[test < float(self.info["data_range_max"])]
    test = test[:int(int(self.info["batch_size"])/int(self.info["div"])) * int(int(self.info["batch_size"])/int(self.info["div"]))]
    x = test.reshape(int(int(self.info["batch_size"])/int(self.info["div"])), int(int(self.info["batch_size"])/int(self.info["div"])))
    '''
    #import pdb
    #pdb.set_trace()
    x = torch.exp(x /t)

    # select half of the data points
    #if(self.select_feature):
    #    x = x[0:int(x.shape[0] * float(self.info["feature_points"])),0:int(x.shape[1] * float(self.info["feature_points"]))]

    # Put positive pairs on the diagonal

    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    
    if(x.size()[0] % 2 != 0):
        idx[x.size()[0] - 1] -= 1


    x = x[idx]

    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))


    log_value = -torch.log(x.mean())
    #print(log_value)
    #import pdb
    #pdb.set_trace()
    return log_value

