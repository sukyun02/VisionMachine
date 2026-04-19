"""
Loss Functions
- Standard CrossEntropy
- CutMix-aware CrossEntropy
- Super-Class Hierarchical Loss with Superclass-Aware Label Smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cifar100 import FINE_TO_COARSE


# ---------------------------------------------------------------------------
# Super-class projection layer (no learnable params)
# ---------------------------------------------------------------------------

def build_fine_to_coarse_matrix(device='cpu') -> torch.Tensor:
    """
    Returns a (20, 100) binary matrix M where M[c, f] = 1
    if fine class f belongs to coarse class c.
    Used to aggregate fine logits → coarse logits.
    """
    M = torch.zeros(20, 100)
    for fine, coarse in enumerate(FINE_TO_COARSE):
        M[coarse, fine] = 1.0
    return M.to(device)


def build_sc_aware_soft_targets(epsilon: float, intra_ratio: float) -> torch.Tensor:
    """
    Builds a (100, 100) soft target matrix for Superclass-Aware Label Smoothing.

    For target class y belonging to superclass c (with n_intra sibling classes):
      - p(y)                         = 1 - epsilon
      - p(i | same superclass, i!=y) = epsilon * intra_ratio / (n_intra - 1)
      - p(i | different superclass)  = epsilon * (1 - intra_ratio) / (100 - n_intra)

    Args:
        epsilon    : total smoothing mass (e.g. 0.1)
        intra_ratio: fraction of epsilon distributed within the same superclass
                     (e.g. 0.5 → half of smoothing stays inside the super-class)

    Returns:
        Tensor of shape (100, 100), rows sum to 1.
    """
    N = 100
    matrix = torch.zeros(N, N)

    # Group fine classes by their superclass
    sc_groups: dict[int, list[int]] = {}
    for fine, coarse in enumerate(FINE_TO_COARSE):
        sc_groups.setdefault(coarse, []).append(fine)

    for y in range(N):
        sc = FINE_TO_COARSE[y]
        intra_classes = sc_groups[sc]
        n_intra = len(intra_classes)
        n_inter = N - n_intra

        # Inter-class (different superclass): uniform small probability
        inter_prob = epsilon * (1.0 - intra_ratio) / n_inter if n_inter > 0 else 0.0
        matrix[y] = inter_prob

        # Intra-class siblings: boosted probability
        intra_prob = epsilon * intra_ratio / (n_intra - 1) if n_intra > 1 else 0.0
        for i in intra_classes:
            if i != y:
                matrix[y, i] = intra_prob

        # Ground-truth class: 1 - epsilon
        matrix[y, y] = 1.0 - epsilon

    return matrix  # (100, 100)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class HierarchicalLoss(nn.Module):
    """
    L_total = (1 - lam_coarse) * CE_fine + lam_coarse * CE_coarse

    Fine-level CE uses Superclass-Aware Label Smoothing:
      - A fraction `intra_ratio` of the smoothing epsilon is distributed
        among sibling classes within the same superclass.
      - Set epsilon=0.0 to disable label smoothing entirely.

    Args:
        lam_coarse  : weight on the coarse-level CE term (default 0.4).
        epsilon     : total label smoothing mass (default 0.1).
        intra_ratio : fraction of epsilon kept within the superclass (default 0.5).

    For CutMix batches supply labels_a, labels_b, lam.
    """

    def __init__(
        self,
        lam_coarse:  float = 0.4,
        epsilon:     float = 0.1,
        intra_ratio: float = 0.5,
    ):
        super().__init__()
        self.lam_coarse  = lam_coarse
        self.epsilon     = epsilon
        self.intra_ratio = intra_ratio

        self._M           = None   # (20, 100) coarse projection, lazy init
        self._soft_matrix = None   # (100, 100) SC-aware soft targets, lazy init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_M(self, device: torch.device) -> torch.Tensor:
        if self._M is None or self._M.device != device:
            self._M = build_fine_to_coarse_matrix(device)
        return self._M

    def _get_soft_matrix(self, device: torch.device) -> torch.Tensor:
        if self._soft_matrix is None or self._soft_matrix.device != device:
            mat = build_sc_aware_soft_targets(self.epsilon, self.intra_ratio)
            self._soft_matrix = mat.to(device)
        return self._soft_matrix

    def _ce_fine(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Fine-level CE with superclass-aware label smoothing.
        Uses soft targets; falls back to standard CE when epsilon == 0.
        """
        if self.epsilon == 0.0:
            return F.cross_entropy(logits, targets)

        soft = self._get_soft_matrix(logits.device)[targets]  # (B, 100)
        log_p = F.log_softmax(logits, dim=-1)                 # (B, 100)
        return -(soft * log_p).sum(dim=-1).mean()

    def _coarse_logits(self, fine_logits: torch.Tensor) -> torch.Tensor:
        M = self._get_M(fine_logits.device)           # (20, 100)
        log_p = F.log_softmax(fine_logits, dim=-1)    # (B, 100)
        return torch.mm(log_p, M.T)                   # (B, 20)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        fine_logits: torch.Tensor,
        labels_a:    torch.Tensor,
        labels_b:    torch.Tensor = None,
        lam:         torch.Tensor = None,
    ) -> torch.Tensor:

        device = fine_logits.device
        mapping = torch.tensor(FINE_TO_COARSE, device=device)

        coarse_logits = self._coarse_logits(fine_logits)
        coarse_a = mapping[labels_a]

        is_cutmix = (labels_b is not None) and (lam is not None) and (lam.item() < 1.0)

        if is_cutmix:
            coarse_b = mapping[labels_b]
            loss_fine = (
                lam       * self._ce_fine(fine_logits, labels_a) +
                (1 - lam) * self._ce_fine(fine_logits, labels_b)
            )
            loss_coarse = (
                lam       * F.nll_loss(coarse_logits, coarse_a) +
                (1 - lam) * F.nll_loss(coarse_logits, coarse_b)
            )
        else:
            loss_fine   = self._ce_fine(fine_logits, labels_a)
            loss_coarse = F.nll_loss(coarse_logits, coarse_a)

        return (1 - self.lam_coarse) * loss_fine + self.lam_coarse * loss_coarse
