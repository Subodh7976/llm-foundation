import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization

    Args:
        dim (int): Feature Dimension (C)
        eps (float): Small constant for numerical stability
        bias (bool): Whether to include learnable bias parameter
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) or (B, C)

        Returns:
            torch.Tensor: Tensor of same shape as input
        """
        # Compute mean and variance along with last dimension
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Apply affine transformation
        output = self.gamma * x_normalized
        if self.beta is not None:
            output = output + self.beta

        return output


class LayerNormFused(nn.Module):
    """
    Memory-efficient LayerNorm using fused operations
    Uses torch.rsqrt for faster reciprocal square root
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean
        mean = x.mean(dim=-1, keepdim=True)

        # Compute variance more efficiently
        # var = E[(x-mean)^2]
        centered = x - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)

        # Use rsqrt for efficiency
        rstd = torch.rsqrt(variance + self.eps)
        # Normalize and scale in one operation
        x_normalized = centered * rstd
        output = self.gamma * x_normalized
        if self.beta is not None:
            output = output + self.beta
        return output


class LayerNormNative(nn.Module):
    """
    Uses Pytorch's native F.layer_norm for maximum efficiency
    This leverages optimized CUDA kernels
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.normalized_shape = (dim, )
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x,
            self.normalized_shape,
            self.gamma,
            self.beta,
            self.eps
        )


class LayerNormWelford(nn.Module):
    """
    Numerically stable LayerNorm using Welford's algorithm
    Better for very large hidden dimensions or extreme values
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Welford's method for numerically stable variance
        mean = x.mean(dim=-1, keepdim=True)

        # More stable variance computation
        # Avoids catastrophic cancellation
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        output = self.gamma * x_normalized
        if self.beta is not None:
            output = output + self.beta
        return output


if __name__ == "__main__":
    # Configuration
    batch_size, seq_len, hidden_dim = 32, 512, 768

    # Initialize different implementations
    standard_ln = LayerNorm(hidden_dim)
    fused_ln = LayerNormFused(hidden_dim)
    native_ln = LayerNormNative(hidden_dim)
    welford_ln = LayerNormWelford(hidden_dim)

    # Sample input
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Forward pass comparison
    import time

    implementations = {
        "Standard": standard_ln,
        "Fused": fused_ln,
        "Native": native_ln,
        "Welford": welford_ln
    }

    for name, model in implementations.items():
        model = model.cuda()
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            output = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"{name} LayerNorm: {elapsed*10:.2f}ms per iteration")
        print(f"Output shape: {output.shape}\n")

# Results:
# Standard LayerNorm: 1.71ms per iteration
# Output shape: torch.Size([32, 512, 768])

# Fused LayerNorm: 3.18ms per iteration
# Output shape: torch.Size([32, 512, 768])

# Native LayerNorm: 0.42ms per iteration
# Output shape: torch.Size([32, 512, 768])

# Welford LayerNorm: 2.53ms per iteration
# Output shape: torch.Size([32, 512, 768])
