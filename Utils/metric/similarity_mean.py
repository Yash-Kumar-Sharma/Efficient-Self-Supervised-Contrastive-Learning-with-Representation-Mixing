import torch
import torch.nn as nn

def Positive_Negative_Mean(x, device, eps=1e-8):
    # Estimate cosine similarity
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    x = (x @ x.t()) / (n * n.t()).clamp(min=eps)
    #x = torch.exp(x /t)
    
    #import pdb
    #pdb.set_trace()
    
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    pos = x.diag()
    #pos_mat = torch.diag(pos).to(device)
    mask = torch.eye(x.size()[0],x.size()[1]).bool().to(device)
    neg_mat = x.clone()
    neg_mat.masked_fill_(mask,0)
    positive = pos
    negative = torch.sum(neg_mat, axis=0)/(len(x) - 2)

    return positive.mean().tolist(), negative.mean().tolist()

def Modified_Positive_Negative_Mean(x, y, device, eps=1e-8):
    # Estimate cosine similarit
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    n1 = torch.norm(y, p=2, dim=1, keepdim=True)

    x = (x @ y.t()) / (n * n1.t()).clamp(min=eps)
    
    #import pdb
    #pdb.set_trace()
    
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    pos = x.diag()
    #pos_mat = torch.diag(pos).to(device)
    mask = torch.eye(x.size()[0],x.size()[1]).bool().to(device)
    neg_mat = x.clone()
    neg_mat.masked_fill_(mask,0)
    positive = pos
    negative = torch.sum(neg_mat, axis=0)/(len(x) - 2)

    return positive.mean().tolist(), negative.mean().tolist()



