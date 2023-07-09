import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor



class FocalLoss(nn.Module):

    """It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha : float = 0.25
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x, y):
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class AdversarialLoss(nn.Module):
    def __init__(self, gpu, reduction='none', ignore_index=255):
        super(AdversarialLoss, self).__init__()
        self.gpu = gpu
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def forward(self, estimated_img_by_g, estimated_img_by_d, label_map, balanced_target, identifier=None):

        source_tensor = torch.zeros_like(estimated_img_by_d).to(self.gpu, dtype=torch.float)
        S_loss = F.cross_entropy(estimated_img_by_g, label_map, reduction=self.reduction,
                                       ignore_index=self.ignore_index)
        loss_of_disc = self.loss(estimated_img_by_d, source_tensor).squeeze()

        class_balance = balanced_target.to(self.gpu)
        general_loss = loss_of_disc * S_loss * class_balance
        return general_loss.mean()


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.tau = 1


    def forward(self, input_img, target_img, labels):
        output = torch.log_softmax(input_img, dim = 1)
        predicted_labels = torch.softmax(target_img * self.tau, dim = 1)
        loss = (output * predicted_labels).mean(dim = 1)

        loss = loss * labels.float()
        outputs = -torch.mean(loss)

        return outputs
