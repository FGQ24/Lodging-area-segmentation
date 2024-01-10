import torch
import torch.nn as nn
import torch.nn.functional as F


def mutil_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)

    return loss, loss0


class GFocalLoss(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        r"""
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(), 1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label = label[mask]
        scale_factor = (pred - label).abs().pow(self.beta)
        loss = F.binary_cross_entropy(pred, label, reduction='none') * scale_factor
        return loss.sum() / pos_num


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(), 1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label = label[mask]
        focal_weight = self.alpha * (label - pred).abs().pow(self.gamma) * (label > 0.0).float() + (
                1 - self.alpha) * pred.abs().pow(self.gamma) * (label <= 0.0).float()
        loss = F.binary_cross_entropy(pred, label, reduction='none') * focal_weight
        return loss.sum() / pos_num


def mutil_GFocal_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    LOSS = GFocalLoss()
    loss0 = LOSS(d0, labels_v)
    loss1 = LOSS(d1, labels_v)
    loss2 = LOSS(d2, labels_v)
    loss3 = LOSS(d3, labels_v)
    loss4 = LOSS(d4, labels_v)
    loss5 = LOSS(d5, labels_v)
    loss6 = LOSS(d6, labels_v)

    loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)

    return loss, loss0


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)

    return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch


def mutil_soft_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = soft_dice_loss(d0, labels_v)
    loss1 = soft_dice_loss(d1, labels_v)
    loss2 = soft_dice_loss(d2, labels_v)
    loss3 = soft_dice_loss(d3, labels_v)
    loss4 = soft_dice_loss(d4, labels_v)
    loss5 = soft_dice_loss(d5, labels_v)
    loss6 = soft_dice_loss(d6, labels_v)

    loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)

    return loss, loss0
