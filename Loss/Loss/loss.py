import torch

def xent_loss(x, t=0.5, eps=1e-8):
    # Estimate cosine similarity
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    x = (x @ x.t()) / (n * n.t()).clamp(min=eps)
    x = torch.exp(x /t)

    # Put positive pairs on the diagonal
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]

    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    log_value = -torch.log(x.mean())

    return log_value


