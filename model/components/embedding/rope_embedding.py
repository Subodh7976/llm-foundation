import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Standard Rotary Positional Embedding (RoPE)

    Args:
        dim (int): Dimension of each attention head (must be even)
        max_seq_len (int): Maximum sequence length to precompute
        base (int): Base for frequency computation (default = 10000)
        device: Device to store tensors
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        device=None
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Dimensions must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # COmpute frequency for each dimension pair
        # theta_i = base ^(-2i/dim) for i in [0, dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cos and sin for all positions up to max_seq_len
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device=None):
        """Precompute cos and sin values for all positions"""
        # Create position indices [0, 1, 2, ..., seq_len - 1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Compute frequencies: outer product of positions and inverse frequencies
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate to match input dimension
        # Shape: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> tuple:
        """
        Args:
            x (torch.Tensor): Input tensor (used only for sequence length if seq_len is not provided)
            seq_len (int, optional): Sequence length. Defaults to None.

        Returns:
            tuple: Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, device=x.device)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    Implements the rotation operation: [-x2, x1, -x4, x3, ...]
    """
    # Block wise rotation
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    # return torch.cat([-x2, x1], dim=-1)

    # Standard: Element wise rotation
    x1 = x[..., 0::2]  # Even indices: [x1, x3, x5, ...]
    x2 = x[..., 1::2]  # Odd indices: [x2, x4, x6, ...]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None
) -> tuple:
    """
    Apply rotary position embedding to queries and keys

    Args:
        q (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim)
        cos (torch.Tensor): Cosine values of shape (seq_len, head_dim)
        sin (torch.Tensor): Sine values of shape (seq_len, head_dim)
        position_ids (torch.Tensor, optional): Optional position indices. Defaults to None.

    Returns:
        tuple: Tuple of rotated (q, k) tensors
    """
    # Handle position_ids for non-contagious positions
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(2)  # (batch, seq_len, 1, head_dim)
        sin = sin[position_ids].unsqueeze(2)
    else:
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryPositionalEmbeddingFused(nn.Module):
    """
    Optimized RoPE with fused operations and efficient memory usage
    Reduces memory allocations and improves cache locality
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Dimension must be event, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Build Cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache with efficient memory layout"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # Apply scaling for context extension
        t = t / self.scaling_factor

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)

        # Different layout: stack cos and sin for better cache performance
        # Shape (seq_len, dim//2, 2)
        cos_sin = torch.stack([freqs.cos(), freqs.sin()], dim=-1)

        self.register_buffer('cos_sin_cached', cos_sin, persistent=False)

    def forward(self, seq_len: int, device=None) -> tuple:
        """
        Returns cos and sin tensors optimized for rotation
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos_sin = self.cos_sin_cached[:seq_len]

        # Expand to full dimension
        cos = cos_sin[..., 0].repeat_interleave(2, dim=-1)
        sin = cos_sin[..., 1].repeat_interleave(2, dim=-1)

        return cos, sin


def apply_rotary_pos_emb_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple:
    """
    Fused RoPE application with optimized memory access patterns
    Uses in-place operations where possible
    """
    # Reshape for rotation: (batch, seq_len, num_heads, head_dim)
    # -> (batch, seq_len, num_heads, head_dim//2, 2)
    *shape, dim = q.shape
    q_reshaped = q.reshape(*shape[:-1], -1, 2)
    k_reshaped = k.reshape(*shape[:-1], -1, 2)

    # Prepare cos/sin with proper broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Extract real and imaginary parts
    q_real, q_imag = q_reshaped.unbind(-1)
    k_real, k_imag = k_reshaped.unbind(-1)

    # Prepare cos/sin for pair-wise operations
    cos_half = cos[..., ::2]
    sin_half = sin[..., ::2]

    # Apply rotation using complex multiplication formula
    # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    q_rotated_real = q_real + cos_half - q_imag * sin_half
    q_rotated_imag = q_real * sin_half + q_imag * cos_half

    k_rotated_real = k_real + cos_half - k_imag * sin_half
    k_rotated_imag = k_real * sin_half + k_imag * cos_half

    # Stack and reshape back
    q_embed = torch.stack([q_rotated_real, q_rotated_imag], dim=-1).flatten(-2)
    k_embed = torch.stack([k_rotated_real, k_rotated_imag], dim=-1).flatten(-2)

    return q_embed, k_embed


class RotaryPositionalEmbeddingExtended(nn.Module):
    """
    RoPE with context length extension using NTK-aware or YaRN interpolation
    Enables training on short contexts and inference on much longer contexts

    Args:
        dim: Head dimension
        max_seq_len: Maximum sequence length
        base: Base frequency (default = 10000)
        scaling_type: 'linear', 'ntk' or 'yarn'
        scaling_factor: How much to extend context (e.g., 2.0 for 2x)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_type: str = 'linear',
        scaling_factor: float = 1.0,
        alpha: float = 1.0,  # For NTK scaling
        beta: float = 32.0   # For YaRN
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        self.beta = beta

        # Compute inverse frequencies with scaling
        inv_freq = self._compute_inv_freq(
            dim, base, scaling_type, scaling_factor, alpha, beta)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _compute_inv_freq(
        self,
        dim: int,
        base: int,
        scaling_type: str,
        scaling_factor: float,
        alpha: float,
        beta: float
    ) -> torch.Tensor:
        """Compute Inverse frequencies with various scaling strategies"""

        if scaling_type == "linear":
            # Simple linear interpolation
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            # Frequencies stay same, but positions are scaled during cache building

        elif scaling_type == "ntk":
            # NTK-aware scaling: adjust base frequency
            # Extends context by modifying the base
            adjusted_base = base * \
                (scaling_factor * alpha) ** (dim / (dim - 2))
            inv_freq = 1.0 / (adjusted_base **
                              (torch.arange(0, dim, 2).float() / dim))

        elif scaling_type == "yarn":
            # YaRN (Yet Another RoPE extension method)
            # Applies different scaling to different frequencies band
            dim_indices = torch.arange(0, dim, 2).float()

            # High frequency (small wavelength) - compress more
            # Low frequency (large wavelength) - compress less
            wavelengths = 2 * math.pi * base ** (dim_indices / dim)

            # Adaptive scaling based on wavelength
            scale = torch.where(
                wavelengths < beta,
                scaling_factor,   # High freq: full scaling
                torch.ones_like(wavelengths)  # Low freq: no scaling
            )

            inv_freq = scale / (base ** (dim_indices / dim))

        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")

        return inv_freq

    def _build_cache(self, seq_len: int):
        """Build cache with position scaling for linear interpolation"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # Apply linear scaling to positions if using linear interpolation
        if self.scaling_type == 'linear':
            t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


class RotaryPositionalEmbeddingGQA(nn.Module):
    """
    RoPE optimized for Grouped Query Attention (GQA)
    Efficiently handles different numbers of Q and KV heads
    """
    def __init__(
        self,
        dim: int, 
        max_seq_len: int = 2048,
        base: int = 10000,
        num_q_heads: int = 32,
        num_kv_heads: int = 8
    ):
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be event, got {dim}")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, seq_len: int) -> tuple:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )
    

def apply_rotary_pos_emb_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int
) -> tuple:
    """
    Apply RoPE for Grouped Query Attention
    Q shape: (batch, seq_len, num_q_heads, head_dim)
    K shape: (batch, seq_len, num_kv_heads, head_dim)
    """
    # Prepare cos/sin
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Apply to queries (all heads)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    # Apply to keys (fewer heads in GQA)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


if __name__ == "__main__":
    # Configuration
    d_model = 768
    num_heads = 12
    head_dim = 64
    batch_size = 16
    seq_len = 512
    max_seq_len = 2048

    # Example 1: Basic RoPE
    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=max_seq_len)

    # Create sample Q and K tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Get cos and sin for current sequence length
    cos, sin = rope(q, seq_len)
    print(f"Cos shape: {cos.shape}")  # (512, 64)
    print(f"Sin shape: {sin.shape}")  # (512, 64)

    # Apply rotation to Q and K
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Rotated Q shape: {q_rotated.shape}")  # (16, 512, 12, 64)
    print(f"Rotated K shape: {k_rotated.shape}")  # (16, 512, 12, 64)

    # Example 2: Context extension with NTK scaling
    rope_extended = RotaryPositionalEmbeddingExtended(
        dim=head_dim,
        max_seq_len=max_seq_len,
        scaling_type='ntk',
        scaling_factor=2.0  # 2x context extension
    )

    # Can now handle longer sequences
    long_seq_len = 4096
    cos_long, sin_long = rope_extended(long_seq_len)
    print(f"Extended context cos shape: {cos_long.shape}")  # (4096, 64)

    # # Example 3: Full attention layer with RoPE
    # attention_layer = MultiHeadAttentionWithRoPE(
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     head_dim=head_dim,
    #     max_seq_len=max_seq_len
    # )

    # x = torch.randn(batch_size, seq_len, d_model)
    # output = attention_layer(x)
    # print(f"Attention output shape: {output.shape}")  # (16, 512, 768)

    # Example 4: GQA with RoPE
    num_q_heads = 32
    num_kv_heads = 8

    rope_gqa = RotaryPositionalEmbeddingGQA(
        dim=head_dim,
        max_seq_len=max_seq_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads
    )

    q_gqa = torch.randn(batch_size, seq_len, num_q_heads, head_dim)
    k_gqa = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)

    cos_gqa, sin_gqa = rope_gqa(seq_len)
    q_gqa_rot, k_gqa_rot = apply_rotary_pos_emb_gqa(q_gqa, k_gqa, cos_gqa, sin_gqa, num_kv_heads)
    print(f"GQA Q shape: {q_gqa_rot.shape}")  # (16, 512, 32, 64)
    print(f"GQA K shape: {k_gqa_rot.shape}")  # (16, 512, 8, 64)

