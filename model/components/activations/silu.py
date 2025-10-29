import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish (SiLU) activation: x * sigmoid(x)"""

    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta (float): Scaling factor for sigmoid input (default to 1.0)
        """
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (*, features)

        Returns:
            torch.Tensor: Activated tensor, same shape as input
        """
        if self.beta == 1.0:
            # use optimized SiLU when beta = 1
            return torch.nn.functional.silu(x)
        return x * torch.sigmoid(self.beta * x)


class SiLU(nn.Module):
    """Alias for Swish with beta=1 (PyTorch standard name)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


class LearnableSwish(nn.Module):
    """Swish with learnable beta parameter"""

    def __init__(self, init_beta: float = 1.0):
        """
        Args:
            init_beta (float, optional): Initial value for beta. Defaults to 1.0.
        """
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class FusedSwish(nn.Module):
    """Swish with fused operations for reduced memory traffic"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pytorch automatically fuses x * sigmoid(x) in JIT mode
        return x * torch.sigmoid(x)


class HardSwish(nn.Module):
    """Piecewise linear approximation for mobile deployment"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hard Swish: x * ReLU6(x + 3) / 6
        # ~4x faster than Swish, used in MobileNetV3
        return x * torch.nn.functional.relu6(x + 3.0) / 6.0


if __name__ == "__main__":
    # Basic usage
    swish = Swish()
    x = torch.randn(64, 512)
    output = swish(x)  # (64, 512), smooth self-gated values

    # In vision model (EfficientNet style)
    class SwishConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.swish = Swish()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, channels, height, width)
            return self.swish(self.bn(self.conv(x)))

    # Learnable beta experiment
    learnable = LearnableSwish(init_beta=1.0)
    x = torch.randn(32, 256, requires_grad=True)
    loss = learnable(x).mean()
    loss.backward()
    print(f"Learned beta: {learnable.beta.item():.4f}")  # Often stays near 1.0

    # Hard Swish for mobile inference
    hard_swish = HardSwish()
    x_mobile = torch.randn(1, 256, 56, 56)  # Mobile input
    output_mobile = hard_swish(x_mobile)  # 4x faster, minimal accuracy loss
