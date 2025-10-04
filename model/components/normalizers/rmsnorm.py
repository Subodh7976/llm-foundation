import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Args:
        dim (int): Feature Dimension (C)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter, initialized to ones
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input Tensor of shape (B, T, C) or (T, C)

        Returns:
            torch.Tensor: Normalized tensor of same shape as input
        """
        # Compute RMS: sqrt(mean(x^2))
        # keepdim=True preserves dimensions for broadcasting
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale by learnable gain
        x_normalized = x / rms
        return x_normalized * self.gain


# Alternative: Memory-Efficient fused implementation
class RMSNormFused(nn.Module):
    """Optimized RMSNorm with fused operations"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use rsqrt (reciprocal square root) for efficiency
        # rsqrt(x) is faster then 1/sqrt(x)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + self.eps)
        return x_normalized * self.gain


if __name__ == "__main__":
    # Configuration
    batch_size, seq_len, hidden_dim = 32, 128, 768

    # Initialize RMSNorm
    rms_norm = RMSNormFused(dim=hidden_dim, eps=1e-6)

    # Sample input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output = rms_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Learnable parameters: {rms_norm.gain.shape}")

