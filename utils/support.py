"""Support-based scalar transformations for MuZero.

MuZero represents scalar values (rewards, values) as categorical distributions
over a discrete support. This improves learning stability compared to direct
regression.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Transform scalar values to categorical support representation.

    Uses the transformation from MuZero:
    1. Apply signed sqrt transformation: sign(x) * (sqrt(|x| + 1) - 1) + eps * x
    2. Clip to support range [-support_size, support_size]
    3. Distribute probability mass to neighboring integers

    Args:
        x: Scalar tensor of shape (...,)
        support_size: Half the support size (support is [-support_size, support_size])

    Returns:
        Categorical distribution of shape (..., 2 * support_size + 1)
    """
    # Signed sqrt transformation for better gradient flow
    eps = 0.001
    transformed = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

    # Clip to support range
    transformed = torch.clamp(transformed, -support_size, support_size)

    # Shift to [0, 2 * support_size] for indexing
    shifted = transformed + support_size

    # Get floor and ceil indices
    floor_idx = shifted.floor().long()
    ceil_idx = floor_idx + 1

    # Clamp indices to valid range
    floor_idx = torch.clamp(floor_idx, 0, 2 * support_size)
    ceil_idx = torch.clamp(ceil_idx, 0, 2 * support_size)

    # Compute weights for floor and ceil
    ceil_weight = shifted - floor_idx.float()
    floor_weight = 1.0 - ceil_weight

    # Create one-hot-like distribution
    batch_shape = x.shape
    support_dim = 2 * support_size + 1

    # Flatten for scatter
    flat_floor = floor_idx.flatten()
    flat_ceil = ceil_idx.flatten()
    flat_floor_weight = floor_weight.flatten()
    flat_ceil_weight = ceil_weight.flatten()

    # Create output tensor
    output = torch.zeros(flat_floor.numel(), support_dim, device=x.device, dtype=x.dtype)

    # Scatter weights
    output.scatter_add_(1, flat_floor.unsqueeze(1), flat_floor_weight.unsqueeze(1))
    output.scatter_add_(1, flat_ceil.unsqueeze(1), flat_ceil_weight.unsqueeze(1))

    # Reshape back to batch shape
    return output.view(*batch_shape, support_dim)


def support_to_scalar(probs: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Transform categorical support distribution back to scalar values.

    Args:
        probs: Categorical distribution of shape (..., 2 * support_size + 1)
        support_size: Half the support size

    Returns:
        Scalar tensor of shape (...,)
    """
    # Create support values [-support_size, ..., support_size]
    support = torch.arange(
        -support_size, support_size + 1,
        device=probs.device,
        dtype=probs.dtype
    )

    # Compute expected value
    expected = (probs * support).sum(dim=-1)

    # Inverse signed sqrt transformation
    eps = 0.001
    sign = torch.sign(expected)
    # Solve: y = sign(x) * (sqrt(|x| + 1) - 1) + eps * x for x
    # Approximate inverse (exact for small eps)
    abs_expected = torch.abs(expected)
    x = sign * ((abs_expected + 1).square() - 1) / (1 + 2 * eps * (abs_expected + 1))

    return x


def compute_cross_entropy_loss(
    pred_logits: torch.Tensor,
    target_scalar: torch.Tensor,
    support_size: int
) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted logits and target scalar.

    Args:
        pred_logits: Predicted logits of shape (..., 2 * support_size + 1)
        target_scalar: Target scalar values of shape (...,)
        support_size: Half the support size

    Returns:
        Cross-entropy loss (scalar)
    """
    target_probs = scalar_to_support(target_scalar, support_size)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1)
    return loss.mean()
