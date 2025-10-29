import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False
    ):
        """
        Args:
            d_model (int): Input/Output dimension
            d_ff (int): Hidden dimension (typically 8/3 * d_model)
            bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
        """
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=bias)
        self.v = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor, shape (batch, seq_len, d_model)
        """
        # Gate pathway: Swish activation
        gate = torch.nn.functional.silu(self.w(x))
        # Value pathway: Linear transformation
        value = self.v(x)
        # Element-wise gating
        hidden = gate * value
        # Project back to d_model
        return self.w2(hidden)


class SwiGLUFused(nn.Module):
    """SwiGLU with fused W and V projections for efficiency"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False
    ):
        super().__init__()
        # Single linear layer for both W and V
        self.wv = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matmul for W and V
        wv = self.wv(x)  # (batch, seq_len, 2*d_ff)
        # Split into gate and value
        w_out, v_out = wv.chunk(2, dim=-1)
        # Apply SwiGLU
        hidden = torch.nn.functional.silu(w_out) * v_out
        return self.w2(hidden)


class SwiGLUCached(nn.Module):
    """SwiGLU with KV-style caching for autoregressive generation"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False
    ):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=bias)
        self.v = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.cache = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = True,
        total_cache_len: int = None
    ) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.w(x))
        value = self.v(x)
        hidden = gate * value

        if use_cache:
            batch, seq_len, d_ff = hidden.shape
            end_pos = start_pos + seq_len

            if self.cache is None:
                assert total_cache_len is not None, "Must specify total_cache_len on first use."
                self.cache = hidden.new_zeros((batch, total_cache_len, d_ff))
                self.cache_pos = 0

            # Overwrite or fill positions directly
            self.cache[:, start_pos:end_pos, :] = hidden
            self.cache_pos = max(self.cache_pos, end_pos)
            hidden = self.cache[:, :self.cache_pos, :]

        return self.w2(hidden)

    def clear_cache(self):
        self.cache = None


class GeGLU(nn.Module):
    """GELU variant of gated linear unit (used in T5)"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int, 
        bias: bool = False
    ):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=bias)
        self.v = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.gelu(self.w(x), approximate='tanh')
        value = self.v(x)
        return self.w2(gate * value)


if __name__ == "__main__":
    # Basic usage in Transformer FFN
    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int = 4096):
            super().__init__()
            d_ff = int(8 / 3 * d_model)  # ~10922 for LLaMA-style
            self.attention = nn.MultiheadAttention(d_model, num_heads=32)
            self.ffn = SwiGLU(d_model, d_ff)
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, d_model)
            # Pre-norm attention
            x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            # Pre-norm FFN with SwiGLU
            x = x + self.ffn(self.norm2(x))
            return x

    # LLaMA-style configuration
    llama_config = {
        'd_model': 4096,
        'd_ff': 11008,  # Rounded from 8/3 * 4096
        'num_layers': 32,
        'num_heads': 32
    }

    ffn = SwiGLU(
        d_model=llama_config['d_model'],
        d_ff=llama_config['d_ff'],
        bias=False  # LLaMA omits bias
    )

    x = torch.randn(4, 512, 4096)  # (batch, seq_len, d_model)
    output = ffn(x)  # (4, 512, 4096)

    # Fused variant for training efficiency
    fused_ffn = SwiGLUFused(d_model=4096, d_ff=11008)
    output_fused = fused_ffn(x)  # Same output, faster

    # Comparison with standard FFN parameter count
    standard_params = 2 * (4096 * 16384)  # W1 and W2 with d_ff=4*d_model
    swiglu_params = 3 * (4096 * 11008)  # W, V, and W2
    print(f"Standard FFN: {standard_params:,} params")  # 134,217,728
    print(f"SwiGLU FFN: {swiglu_params:,} params")      # 135,266,304

