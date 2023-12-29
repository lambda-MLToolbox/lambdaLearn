import torch
import torch.nn as nn
import torch.nn.functional as F


class Cross_Entropy(nn.Module):
    def __init__(self, use_hard_labels=True, reduction="mean"):
        """
        :param use_hard_labels: Whether to use hard labels in the consistency regularization.
        :param reduction: How to handle the output.
        """
        super(Cross_Entropy, self).__init__()
        self.use_hard_labels = use_hard_labels
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: The output of the model.
        :param targets: The target result.
        """
        targets = targets.long()
        if self.use_hard_labels:
            log_pred = F.log_softmax(logits, dim=-1)
            return F.nll_loss(log_pred, targets, reduction=self.reduction)
        else:
            log_pred = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets * log_pred, dim=1)
            if self.reduction == "mean":
                nll_loss = nll_loss.mean()
            elif self.reduction == "sum":
                nll_loss = nll_loss.sum()
            return nll_loss
