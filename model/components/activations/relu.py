import torch
import torch.nn as nn


class ReLU(nn.Module):
    """Standard ReLU activation: max(0, x)"""
    
    def __init__(self, inplace: bool = False):
        """
        Args:
            inplace: If True, modify input tensor directly (saves memory)
        """
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (*, features)
        Returns:
            Activated tensor, same shape as input
        """
        return torch.relu(x, inplace=self.inplace)
    

class FusedReLU(nn.Module):
    """ReLU fused bias addition"""
    
    def __init__(self, num_features: int, inplace: bool = False):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fuse bias add with ReLU to reduce memory traffic
        return torch.relu(x + self.bias, inplace=self.inplace)


class LeakyReLU(nn.Module):
    """Leaky ReLU tor prevent dead neurons"""
    
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        """
        Args:
            negative_slope (float, optional): Slope for negative inputs. Defaults to 0.01.
            inplace (bool, optional): Memory optimization flag. Defaults to False.
        """
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(
            x, negative_slope=self.negative_slope, inplace=self.inplace
        )


class PReLU(nn.Module):
    """Parametric ReLU with learnable negative slope"""
    
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """
        Args:
            num_parameters (int, optional): 1 for shared slope, or feature count for per-channel. Defaults to 1.
            init (float, optional): Initial value for negative slope parameter. Defaults to 0.25.
        """
        self.weight = nn.Parameter(torch.full((num_parameters,), init))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.prelu(x, self.weight)


if __name__ == "__main__":
    # Basic usage
    relu = ReLU()
    x = torch.randn(32, 512)  # (batch_size, features)
    output = relu(x)  # (32, 512), negative values zeroed

    # In Transformer FFN
    class StandardFFN(nn.Module):
        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff)
            self.activation = ReLU()
            self.w2 = nn.Linear(d_ff, d_model)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, d_model)
            return self.w2(self.activation(self.w1(x)))

    # Edge case: All negative inputs
    x_neg = torch.randn(16, 256).neg()
    output_neg = relu(x_neg)  # All zeros

    # Leaky variant for dead neuron mitigation
    leaky = LeakyReLU(negative_slope=0.1)
    output_leaky = leaky(x_neg)  # 0.1 * x_neg instead of zeros
