"""Utility layers and functions for domain-adversarial training."""

import torch
import torch.nn as nn


class GradientReversalLayer(nn.Module):
    """
    Gradient reversal layer for domain adversarial training.
    
    During forward pass, returns input unchanged.
    During backward pass, returns gradient with reversed (negated) sign.
    
    This forces the encoder to learn features that are non-discriminative
    with respect to the auxiliary domain classifier.
    
    Reference: Ganin & Lempitsky (2015) - Unsupervised Domain Adaptation by Backpropagation
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Multiplier for gradient magnitude. Higher values increase 
                   domain adversarial pressure. Typical range: [0.1, 1.0].
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns input unchanged."""
        return x
    
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass reverses and scales gradient."""
        return -self.alpha * grad_output


class ReverseLayerF(torch.autograd.Function):
    """
    Functional gradient reversal using autograd.Function for efficiency.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def reverse_gradient(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Apply gradient reversal to tensor.
    
    Args:
        x: Input tensor
        alpha: Gradient reversal strength
    
    Returns:
        Tensor with negated gradients
    """
    return ReverseLayerF.apply(x, alpha)
