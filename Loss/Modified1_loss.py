import torch

def xent_loss(x, y, t=0.5, eps=1e-8):
    # Estimate cosine similarity
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    n1 = torch.norm(y, p=2, dim=1, keepdim=True)

    x = (x @ y.t()) / (n * n1.t()).clamp(min=eps)
    
    x = torch.exp(x /t)
    #x = x[0:int(x.shape[0]/2),0:int(x.shape[1]/2)] 
    # Put positive pairs on the diagonal
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    log_value = -torch.log(x.mean())

    return log_value
