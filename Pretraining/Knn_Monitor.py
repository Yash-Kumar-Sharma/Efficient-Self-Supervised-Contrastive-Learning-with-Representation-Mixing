import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm

# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
class Knn_Monitor():
    
    def __init__(self, config) -> None:     
        self.k = config.dataset.knn_k
        self.c = config.dataset.num_classes
        self.batch_size = config.dataset.batch_size

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])        
        memory_data = torchvision.datasets.CIFAR10(root=config.dataset.data_dir, train=True, transform=test_transform, download=True)
        self.memory_data_loader = torch.utils.data.DataLoader(memory_data, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_data = torchvision.datasets.CIFAR10(root=config.dataset.data_dir, train=False, transform=test_transform, download=True)
        self.test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # test using a knn monitor
    def test(self, net, k=200, t=0.1, hide_progress=False):
        net.eval()
        classes = len(self.memory_data_loader.dataset.classes)
        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(self.memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
                feature = net(data.cuda(non_blocking=True))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(self.memory_data_loader.dataset.targets, device=feature_bank.device)
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.test_data_loader, desc='kNN', disable=hide_progress)
            for data, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)
                
                pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes, k, t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})
        return total_top1 / total_num * 100

    # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels
