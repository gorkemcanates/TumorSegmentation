import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.functional as MF
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(a):
    plt.figure()
    plt.imshow(a)

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1))
        acc = (torch.sum(preds == y)) / (preds.size(0) * preds.size(1) *
                                         preds.size(2))
        return acc.detach().cpu().numpy().item()


class IoU(nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, yhat, y, eps=1e-7):
        B = y.shape[0]
        target = y.unsqueeze(1)

        if self.num_classes == 1:
            target_1_hot = torch.eye(self.num_classes + 1)[target.type(
                torch.LongTensor).squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            target_1_hot_f = target_1_hot[:, 0:1, :, :]
            target_1_hot_s = target_1_hot[:, 1:2, :, :]
            target_1_hot = torch.cat([target_1_hot_s, target_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(yhat)
            neg_prob = 1 - pos_prob
            probas1 = torch.cat([pos_prob, neg_prob], dim=1)
            probas1 = torch.argmax(probas1, dim=1, keepdim=True)
            probas = torch.cat([1 - probas1, probas1], dim=1)

        else:

            target_1_hot = torch.eye(self.num_classes)[target.type(
                torch.LongTensor).squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            probas1 = F.softmax(yhat, dim=1)

            probas1 = torch.argmax(probas1, dim=1)
            probas = torch.zeros_like(yhat)
            probas[probas1 == 1, 1] = 1
            probas[probas1 == 2, 2] = 1
            probas[probas.sum(dim=1) != 1, 0] = 1

        target_1_hot = target_1_hot.type(yhat.type())

        # intersection = probas * target_1_hot
        # union = probas + target_1_hot - intersection
        #
        # # BCHW -> BC
        # intersection = intersection.view(B, self.num_classes, -1).sum(2)
        # union = union.view(B, self.num_classes, -1).sum(2)
        #
        # jacc = (intersection / (union + eps))
        # jacc_loss = torch.mean(jacc.mean(1))


        dims = (0,) + tuple(range(2, y.ndimension()))
        intersection = torch.sum(probas * target_1_hot, dims)
        cardinality = torch.sum(probas + target_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()

        return jacc_loss.detach().cpu().numpy().item()


class DiceScore(nn.Module):
    def __init__(self, num_classes=3, smooth=0, reduction='mean'):
        super(DiceScore, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, yhat, y, eps=1e-7):
        B = y.shape[0]
        target = y.unsqueeze(1)

        if self.num_classes == 1:
            target_1_hot = torch.eye(self.num_classes + 1)[target.type(
                torch.LongTensor).squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            target_1_hot_f = target_1_hot[:, 0:1, :, :]
            target_1_hot_s = target_1_hot[:, 1:2, :, :]
            target_1_hot = torch.cat([target_1_hot_s, target_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(yhat)
            neg_prob = 1 - pos_prob
            probas1 = torch.cat([pos_prob, neg_prob], dim=1)
            probas1 = torch.argmax(probas1, dim=1, keepdim=True)
            probas = torch.cat([1 - probas1, probas1], dim=1)

        else:

            target_1_hot = torch.eye(self.num_classes)[y.type(
                torch.LongTensor).squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            probas1 = F.softmax(yhat, dim=1)
            probas1 = torch.argmax(probas1, dim=1)
            probas = torch.zeros_like(yhat)
            probas[probas1 == 1, 1] = 1
            probas[probas1 == 2, 2] = 1
            probas[probas.sum(dim=1) != 1, 0] = 1

        target_1_hot = target_1_hot.type(yhat.type())
        dims = (0,) + tuple(range(2, y.ndimension()))
        intersection = torch.sum(probas * target_1_hot, dims)
        cardinality = torch.sum(probas + target_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        # intersection = probas * target_1_hot
        # cardinality = probas + target_1_hot
        #
        # intersection = intersection.view(B, self.num_classes, -1).sum(2)
        # cardinality = cardinality.view(B, self.num_classes, -1).sum(2)
        # dice_loss = ((2. * intersection + self.smooth) /
        #              (cardinality + eps + self.smooth))
        # dice_loss = torch.mean(dice_loss.mean(1))

        return dice_loss.detach().cpu().numpy().item()


class FBeta(nn.Module):
    def __init__(self, in_class, beta=1, eps=1e-6):
        super().__init__()

        self.in_class = in_class
        self.beta = beta
        self.eps = eps

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = torch.argmax(yhat, dim=1)
        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FN = torch.sum((preds == 0) * (y == self.in_class))
        FP = torch.sum((preds == self.in_class) * (y == 0))

        f_beta = ((1 + self.beta ** 2) * TP) / \
                 ((1 + self.beta ** 2) * TP + (self.beta ** 2) * FN + FP)
        return f_beta.detach().cpu().numpy().item()


class PostIoU(nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super(PostIoU, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, yhat, y):
        probas1 = F.softmax(yhat, dim=1)
        probas = torch.argmax(probas1, dim=1)

        mask1 = torch.where(y == 1, 1, 0).type(torch.uint8)
        mask2 = torch.where(y == 2, 1, 0).type(torch.uint8)
        kernel = np.ones((3, 3))
        erosion1 = []
        dilation1 = []
        erosion2 = []
        dilation2 = []
        for i in range(len(y)):
            m1 = mask1[i, :, :]
            m2 = mask2[i, :, :]
            erosion1.append(cv2.erode(m1.numpy(), kernel, iterations=1))
            dilation1.append(cv2.dilate(m1.numpy(), kernel, iterations=1))
            erosion2.append(cv2.erode(m2.numpy(), kernel, iterations=1))
            dilation2.append(cv2.dilate(m2.numpy(), kernel, iterations=1))

        erosion1 = torch.tensor(erosion1, dtype=torch.uint8)
        dilation1 = torch.tensor(dilation1, dtype=torch.uint8)
        erosion2 = torch.tensor(erosion2, dtype=torch.uint8)
        dilation2 = torch.tensor(dilation2, dtype=torch.uint8)

        erodedGroundtruth = torch.zeros_like(y, dtype=torch.uint8)
        erodedGroundtruth[erosion1 == 1] = 1
        erodedGroundtruth[erosion2 == 1] = 2

        dilatedGroundtruth = torch.zeros_like(y, dtype=torch.uint8)
        dilatedGroundtruth[dilation1 == 1] = 1
        dilatedGroundtruth[dilation2 == 1] = 2

        intersection = torch.where(torch.logical_and(dilatedGroundtruth == probas, dilatedGroundtruth != 0), 1, 0)
        intersectionCount = torch.count_nonzero(intersection, dim=(1, 2))

        union = torch.where(torch.logical_or(erodedGroundtruth != 0, probas != 0), 1, 0)
        unionCount = torch.count_nonzero(union, dim=(1, 2))

        score = intersectionCount / unionCount
        return score.mean().detach().cpu().numpy().item()


class MaskMeanMetric(nn.Module):
    def __init__(self, in_channels=3, reduction='mean'):
        super(MaskMeanMetric, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

    def forward(self, yhat, y):
        probas1 = F.softmax(yhat, dim=1)
        probas = torch.argmax(probas1, dim=1)

        class1Color = (255, 0, 0)
        class2Color = (0, 255, 0)

        groundtruth_maskRGB = np.zeros((y.numpy().shape[0], y.numpy().shape[1], y.numpy().shape[2], 3), dtype=np.uint8)
        groundtruth_maskRGB[y.numpy() == 1] = class1Color
        groundtruth_maskRGB[y.numpy() == 2] = class2Color
        groundtruth_maskRGB = torch.tensor(groundtruth_maskRGB, dtype=torch.uint8)

        predicted_maskRGB = np.zeros((probas.numpy().shape[0], probas.numpy().shape[1], probas.numpy().shape[2], 3), dtype=np.uint8)
        predicted_maskRGB[probas.numpy() == 1] = class1Color
        predicted_maskRGB[probas.numpy() == 2] = class2Color
        predicted_maskRGB = torch.tensor(predicted_maskRGB, dtype=torch.uint8)

        maskr = torch.clone(groundtruth_maskRGB[:, :, :, 0])
        maskg = torch.clone(groundtruth_maskRGB[:, :, :, 1])
        kernel = np.ones((3, 3))

        erosionr = []
        dilationr = []
        erosiong = []
        dilationg = []
        for i in range(len(y)):
            m1 = maskr[i, :, :]
            m2 = maskg[i, :, :]
            erosionr.append(cv2.erode(m1.numpy(), kernel, iterations=1))
            dilationr.append(cv2.dilate(m1.numpy(), kernel, iterations=1))
            erosiong.append(cv2.erode(m2.numpy(), kernel, iterations=1))
            dilationg.append(cv2.dilate(m2.numpy(), kernel, iterations=1))

        erosionr = torch.tensor(erosionr, dtype=torch.uint8)
        dilationr = torch.tensor(dilationr, dtype=torch.uint8)
        erosiong = torch.tensor(erosiong, dtype=torch.uint8)
        dilationg = torch.tensor(dilationg, dtype=torch.uint8)
        eroded_gt = torch.clone(groundtruth_maskRGB)
        eroded_gt[:, :, :, 0] = erosionr
        eroded_gt[:, :, :, 1] = erosiong

        dilated_gt = torch.clone(groundtruth_maskRGB)
        dilated_gt[:, :, :, 0] = dilationr
        dilated_gt[:, :, :, 1] = dilationg

        y = torch.where(y > 0, 255, 0)
        eroded_gt = torch.where(eroded_gt > 0, 255, 0)
        dilated_gt = torch.where(dilated_gt > 0, 255, 0)
        probas = torch.where(probas > 0, 255, 0)
        totalfp = torch.zeros((len(yhat), ))
        totalfn = torch.zeros((len(yhat), ))
        totaltp = torch.zeros((len(yhat), ))
        totaltn = torch.zeros((len(yhat), ))
        fpimg = torch.where(torch.logical_and(dilated_gt == 0, predicted_maskRGB == 255), 255, 0)
        fp = torch.count_nonzero(fpimg, dim=(1, 2, 3))
        totalfp = totalfp + fp

        fnimg = torch.where(torch.logical_and(eroded_gt == 255, predicted_maskRGB == 0), 255, 0)
        fn = torch.count_nonzero(fnimg, dim=(1, 2, 3))
        totalfn = totalfn + fn

        tp = (torch.count_nonzero(predicted_maskRGB, dim=(1, 2, 3)) - fp)
        totaltp = totaltp + tp

        area = torch.ones((len(y), )) * (predicted_maskRGB.shape[1] * predicted_maskRGB.shape[2])
        tn = area - (fp + fn + tp)
        totaltn = totaltn + tn

        sensitivity = totaltp / (totaltp + totalfn)
        specifity = totaltn / (totaltn + totalfp)

        score = (sensitivity + specifity) / 2

        return score.mean().detach().cpu().numpy().item()
