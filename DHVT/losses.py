# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import pdb


def build_superclass_smooth_matrix(fine_to_coarse, num_classes, smoothing=0.1, intra_ratio=0.8):
    """Build a (num_classes, num_classes) smoothing matrix for superclass-aware label smoothing.

    smooth_matrix[i] is the full target distribution when ground truth is class i.
    Same-superclass classes get more smoothing mass than cross-superclass classes.

    Args:
        fine_to_coarse: list mapping fine class index → coarse class index
        num_classes: number of fine classes
        smoothing: total smoothing mass
        intra_ratio: fraction of smoothing mass allocated to same-superclass classes
    Returns:
        Tensor of shape (num_classes, num_classes)
    """
    ftc = torch.tensor(fine_to_coarse, dtype=torch.long)
    smooth_matrix = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        same_sc = (ftc == ftc[i])
        n_intra = same_sc.sum().item() - 1  # exclude self
        n_inter = num_classes - n_intra - 1

        intra_mass = smoothing * intra_ratio
        inter_mass = smoothing * (1 - intra_ratio)

        smooth_matrix[i] = inter_mass / n_inter
        smooth_matrix[i, same_sc] = intra_mass / n_intra
        smooth_matrix[i, i] = 1.0 - smoothing
    return smooth_matrix


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    Optionally adds auxiliary superclass classification loss.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float,
                 fine_to_coarse=None, coarse_loss_weight: float = 0.5):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.coarse_loss_weight = coarse_loss_weight

        # Build fine-to-coarse aggregation matrix (num_fine x num_coarse)
        if fine_to_coarse is not None:
            self.register_buffer('fine_to_coarse', torch.tensor(fine_to_coarse, dtype=torch.long))
            num_fine = len(fine_to_coarse)
            num_coarse = max(fine_to_coarse) + 1
            agg = torch.zeros(num_fine, num_coarse)
            for i, c in enumerate(fine_to_coarse):
                agg[i, c] = 1.0
            self.register_buffer('agg_matrix', agg)
        else:
            self.fine_to_coarse = None
            self.agg_matrix = None

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output.
                When superclass head is active, an extra coarse_logits tensor is appended.
            labels: the labels for the base criterion
        """
        outputs_kd = None
        coarse_logits = None
        if not isinstance(outputs, torch.Tensor):
            if len(outputs) == 3:
                # distillation + superclass: (fine, dist, coarse)
                outputs, outputs_kd, coarse_logits = outputs
            elif self.fine_to_coarse is not None:
                # superclass only: (fine, coarse)
                outputs, coarse_logits = outputs
            else:
                # distillation only: (fine, dist)
                outputs, outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)

        # Coarse (superclass) loss
        coarse_loss = torch.tensor(0.0, device=outputs.device)
        if coarse_logits is not None and self.fine_to_coarse is not None:
            if labels.dim() == 1:
                # Hard labels (no mixup)
                coarse_labels = self.fine_to_coarse[labels]
                coarse_loss = F.cross_entropy(coarse_logits, coarse_labels)
            else:
                # Soft labels (mixup) — aggregate fine soft targets to coarse
                coarse_targets = labels @ self.agg_matrix
                coarse_loss = -torch.sum(coarse_targets * F.log_softmax(coarse_logits, dim=1)) / coarse_logits.size(0)

        if self.distillation_type == 'none':
            if self.fine_to_coarse is not None:
                return (1 - self.coarse_loss_weight) * base_loss + self.coarse_loss_weight * coarse_loss
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        distill_loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        if self.fine_to_coarse is not None:
            loss = (1 - self.coarse_loss_weight) * distill_loss + self.coarse_loss_weight * coarse_loss
        else:
            loss = distill_loss
        return loss

    

def DiversityLoss(inputs):
    B, depth, head, C = inputs.shape
    thr = 0.0
    label = (1-thr) * torch.eye(head, head).expand(B, depth, head, head).cuda()
    inputs = F.normalize(inputs, dim=-1)
    cos_mat = inputs @ inputs.transpose(-2, -1)
    label = label + thr
    loss = 0.2*(cos_mat-label).sum()/(B*head*head*depth)
    
    return loss
