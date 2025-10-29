import torch
import torch.nn as nn
import math


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation"""

    def __init__(self, approximate: str = 'none'):
        """
        Args:
            approximate (str, optional): 'none' for exact, 'tanh' for tanh approximation. Defaults to 'none'.
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input Tensor, shape (* , features)

        Returns:
            torch.Tensor: Activated tensor, same shape as input
        """
        return torch.nn.functional.gelu(x, approximate=self.approximate)


class GELUExact(nn.Module):
    """Exact GELU using error function"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELUTanh(nn.Module):
    """Fast tanh approximation of GELU"""

    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        self.coeff = 0.044715

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ~2x faster than exact GELU, <0.1% accuracy loss
        inner = self.sqrt_2_over_pi * (x + self.coeff * torch.pow(x, 3))
        return 0.5 * x * (1.0 + torch.tanh(inner))


class GELUSigmoid(nn.Module):
    """Sigmoid approximation of GELU (fastest)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ~3x faster than exact, slightly lower accuracy
        return x * torch.sigmoid(1.702 * x)


class FusedGELU(nn.Module):
    """GELU approximation of GELU (fastest)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        approximate: str = "tanh"
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ~3x faster than exact, slightly lower accuracy
        return x * torch.sigmoid(1.702 * x)


if __name__ == "__main__":
    # Basic usage
    gelu = GELU(approximate='tanh')
    x = torch.randn(32, 768)  # BERT hidden size
    output = gelu(x)  # (32, 768), smooth non-linearity

    # In Transformer FFN (BERT/GPT-2 style)
    class GELUFFN(nn.Module):
        def __init__(self, d_model: int = 768, d_ff: int = 3072):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff)
            self.gelu = GELU(approximate='tanh')
            self.w2 = nn.Linear(d_ff, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, d_model)
            return self.w2(self.gelu(self.w1(x)))

    # Comparison of variants
    x = torch.randn(1000, 512)
    exact = GELUExact()(x)
    tanh_approx = GELUTanh()(x)
    sigmoid_approx = GELUSigmoid()(x)

    # Maximum absolute error
    print(f"Tanh error: {(exact - tanh_approx).abs().max():.6f}")  # ~0.0001
    # ~0.001
    print(f"Sigmoid error: {(exact - sigmoid_approx).abs().max():.6f}")
