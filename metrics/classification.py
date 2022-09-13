import torch
from torch import nn


class ErrorRate_T1(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, yhat, y):
        probs = self.softmax(yhat).detach()
        preds = torch.argmax(probs, dim=1)
        target = torch.argmax(y.squeeze(2), dim=1)

        acc = torch.sum(preds == target) / yhat.size(0)
        err = 1 - acc.cpu().numpy().item()
        return err * 100


class ErrorRate_T5(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.k = k

    def forward(self, yhat, y):
        probs = self.softmax(yhat).detach()
        target = torch.argmax(y.squeeze(2), dim=1)
        _, preds = probs.topk(k=self.k, dim=1)
        target = target.unsqueeze(1).expand(preds.shape[0], preds.shape[1])
        acc = torch.sum(preds == target) / yhat.size(0)
        err = 1 - acc.cpu().numpy().item()
        return err * 100
