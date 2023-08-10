import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu
from kornia.morphology import dilation

from collections import defaultdict


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self,
                input,      # (1, 3, 384, 384)
                target,     # (1, 384, 384) <- binary
                it):

        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class BoundaryBCELoss(nn.Module):
    def __init__(self, iterations: int = 5):
        super().__init__()
        self.kernel = torch.Tensor([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]
                                    ])
        self.iterations = iterations

    def forward(self,
                hand_mask: torch.Tensor,
                object_mask: torch.Tensor,
                target: torch.Tensor,
                ):
        """
        Expect input is probability masks of hand and object
        hand_mask:      (N, 1, 384, 384)
        object_mask:    (N, 1, 384, 384)
        """
        if self.kernel.device != hand_mask.device:
            self.kernel = self.kernel.to(hand_mask.device, non_blocking=True)

        for _ in range(self.iterations):
            # Dilate the mask.
            hand_mask = dilation(hand_mask, self.kernel)
            object_mask = dilation(object_mask, self.kernel)

        # Create the contact boundary mask.
        # Use multiply to mimic the logical_and operator but differentiable
        contact_boundary = hand_mask * object_mask # (N, 1, 384, 384)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(contact_boundary, target)

        return loss

class CBBCELoss(nn.Module):
    def __init__(self, iterations: int = 5):
        super().__init__()
        self.kernel = torch.Tensor([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]
                                    ])
        self.iterations = iterations

    def forward(self,
                mask: torch.Tensor,
                target: torch.Tensor,
                ):
        """
        Expect input is probability masks of hand and object
        hand_mask:      (N, 1, 384, 384)
        object_mask:    (N, 1, 384, 384)
        """
        if self.kernel.device != mask.device:
            self.kernel = self.kernel.to(mask.device, non_blocking=True)

        # Create the contact boundary mask.
        # Use multiply to mimic the logical_and operator but differentiable
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(mask, target)

        return loss

class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE()

    def compute(self, data, it, weight=0):
        """
        Compute multiple losses for a batch of data
        but only total_loss is used for backprop
        Input:
            logit_i (i=0, 1, 2) -> logit for each object in batch
                (1, 3, H, W) if there is 2 objects, one logit is for aggregated (in STM paper)
                (1, 2, H, W) if there is single object

            cls_gt (1, H, W) where each pixel value is object index

        loss_i (i=0, 1, 2 ...) -> loss for each object in batch
        total_loss := sum(loss_i)
        hide_iou/i
        hide_iou/u
        hide_iou/sec_i
        hide_iou_sec_u

        """
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape # (N, T, 1, H, W)
        selector = data.get('selector', None)

        for i in range(1, s):
            # Loop over frame in a sequence. Start from 1 since 0 has GT provided

            for j in range(b):
                # Loop over each data in a batch
                # since not every entry has the second object

                # 0th mask -> aggregated mask
                # 1st mask -> first object mask
                # 2nd mask -> second object mask

                if selector is not None and selector[j][1] > 0.5:
                    # Sequence has second object, evaluate all 3 logits masks
                    loss, p = self.bce(data['logits_%d'%i][j: j+1], # (1, 3, 384, 384)
                                       data['cls_gt'][j: j+1, i],   # (1, 384, 384)
                                       it)
                else:
                    # Otherwise, only evaluate first 2 logits masks
                    loss, p = self.bce(data['logits_%d'%i][j: j+1, :2], # (1, 2, 384, 384)
                                       data['cls_gt'][j: j+1, i],       # (1, 384, 384)
                                       it)

                losses['loss_%d'%i] += loss / b # avg loss of each frame in sequences
                losses['p'] += p / b / (s-1) # p is topk ratio

            losses['total_loss'] += losses['loss_%d'%i]

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i] > 0.5,
                                                         data['gt'][:, i] > 0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i] > 0.5,
                                                             data['sec_gt'][:, i] > 0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u

        return losses


class CombineLossComputer:
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()
        self.boundary_bce = BoundaryBCELoss()

    def compute(self, data, it, weight = 0.1):
        """
        There are two losses:
            - BootstrappedBCE for object segmentation
            - BoundaryBCE for boundary segmentation
        """
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape # (N, T, 1, H, W)
        selector = data.get('selector', None)

        # Compute BoostrapedCE loss over masks
        for i in range(1, s):

            # Loop over frame in a sequence. Start from 1 since 0 has GT provided
            for j in range(b):
                # Loop over each data in a batch
                # since not every entry has the second object

                # 0th mask -> aggregated mask
                # 1st mask -> first object mask
                # 2nd mask -> second object mask

                if selector is not None and selector[j][1] > 0.5:
                    # Sequence has second object, evaluate all 3 logits masks
                    loss, p = self.bce(data['logits_%d'%i][j: j+1], # (1, 3, 384, 384)
                                       data['cls_gt'][j: j+1, i],   # (1, 384, 384)
                                       it)
                else:
                    # Otherwise, only evaluate first 2 logits masks
                    loss, p = self.bce(data['logits_%d'%i][j: j+1, :2], # (1, 2, 384, 384)
                                       data['cls_gt'][j: j+1, i],       # (1, 384, 384)
                                       it)

                losses['loss_%d'%i] += loss / b # avg loss of each frame in sequences
                losses['p'] += p / b / (s-1) # p is topk ratio

            losses['mask_loss'] += losses['loss_%d'%i]

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i] > 0.5,
                                                         data['gt'][:, i] > 0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i] > 0.5,
                                                             data['sec_gt'][:, i] > 0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u

            # Compute Boundary BCE loss
            losses['boundary_loss'] += self.boundary_bce(data['mask_%d'%i],         # hand mask
                                                         data['sec_mask_%d'%i],     # object mask
                                                         data['boundary_gt'][:, i:i+1])       # boundary GT
        if weight < 0:
            losses['total_loss'] = losses['mask_loss'] + losses['boundary_loss']
        else:
            losses['total_loss'] = (1 - weight) * losses['mask_loss'] + weight * losses['boundary_loss']

        return losses


class CBCombineLossComputer:
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()
        self.boundary_bce = CBBCELoss()

    def compute(self, data, it, weight = 0.1):
        """
        There are two losses:
            - BootstrappedBCE for object segmentation
            - BoundaryBCE for boundary segmentation
        """
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape # (N, T, 1, H, W)
        selector = data.get('selector', None)

        # Compute BoostrapedCE loss over masks
        for i in range(1, s):

            # Loop over frame in a sequence. Start from 1 since 0 has GT provided
            for j in range(b):
                # Loop over each data in a batch
                # since not every entry has the second object

                # 0th mask -> aggregated mask
                # 1st mask -> first object mask
                # 2nd mask -> second object mask

                if selector is not None and selector[j][1] > 0.5:
                    # Sequence has second object, evaluate all 3 logits masks
                    loss, p = self.bce(data['logits_%d'%i][j: j+1], # (1, 3, 384, 384)
                                       data['cls_gt'][j: j+1, i],   # (1, 384, 384)
                                       it)
                else:
                    # Otherwise, only evaluate first 2 logits masks
                    loss, p = self.bce(data['logits_%d'%i][j: j+1, :2], # (1, 2, 384, 384)
                                       data['cls_gt'][j: j+1, i],       # (1, 384, 384)
                                       it)

                losses['loss_%d'%i] += loss / b # avg loss of each frame in sequences
                losses['p'] += p / b / (s-1) # p is topk ratio

            losses['mask_loss'] += losses['loss_%d'%i]

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i] > 0.5,
                                                         data['gt'][:, i] > 0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i] > 0.5,
                                                             data['sec_gt'][:, i] > 0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u

            # Compute Boundary BCE loss
            losses['boundary_loss'] += self.boundary_bce(data['cb_%d'%i],                       # hand mask
                                                         data['boundary_gt'][:, i:i+1])         # boundary GT
        if weight < 0:
            losses['total_loss'] = losses['mask_loss'] + losses['boundary_loss']
        else:
            losses['total_loss'] = (1 - weight) * losses['mask_loss'] + weight * losses['boundary_loss']

        return losses

