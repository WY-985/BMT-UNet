import torch
import torch.nn as nn
class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init__()


    def forward(self, input, target, beta=0.5, weights=None):
        N = target.size(0)
        smooth = 1
        alpha = 1-beta

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        false_neg = target_flat * (1-input_flat)
        flase_pos = (1-target_flat) * input_flat
        loss = (intersection.sum(1) + smooth)/(intersection.sum(1) + alpha*false_neg.sum(1) + beta*flase_pos.sum(1) + smooth)
        loss = 1 - loss.sum()/N
        return loss

