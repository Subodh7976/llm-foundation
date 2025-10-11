import torch
import torch.nn as nn
import math


class ALiBiAttention(nn.Module):
    """
    Multi-head attention with ALiBi (Attention with Linear Biases)

    Adds linearly decreasing biases to attention scores based on distance,
    enabling extrapolation to longer sequences than seen during training
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Compute ALiBi slopes (fixed, not learned)
        self.register_buffer('slopes', self._get_slopes(n_heads))

    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Compute head-specific slopes as geometric sequence.

        Returns:
            slopes: Tensor of shape (n_heads, ) with negative slopes
        """
        # Geometric sequence: 2^(-8/n), 2^(-2*8/n), ..., 2^(-8)
        def get_slopes_power_of_2(n: int) -> torch.Tensor:
            start = 2 ** (-8 / n)
            ratio = start
            return torch.Tensor([start * (ratio ** i) for i in range(n)])

        # Handle non-power-of-2 by interpolation
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)

        # Find closest power of 2
        closest_power = 2 ** math.floor(math.log2(n_heads))
        base_slopes = get_slopes_power_of_2(closest_power)
        # Interpolate additional slopes
        extra_slopes = get_slopes_power_of_2(
            2 * closest_power)[::2][:n_heads - closest_power]
        return torch.cat([base_slopes, extra_slopes])

    def _get_alibi_bias(self, seq_len: int, device) -> torch.Tensor:
        """
        Create ALiBi bias matrix: m * [-(i-1), ..., -2, -1, 0] for each head.

        Args:
            seq_len (int): Sequence length
            device: Device to create tensor on

        Returns:
            torch.Tensor: Tensor of shape (n_heads, seq_len, seq_len)
        """
        # Create distance matrix: position_i - position_j
        # For casual attention, only lower triangular part matters
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]

        # Relative distances (negative for past positions)
        relative_position = memory_position - context_position

        # Apply slopes: (n_heads, 1, 1) * (1, seq_len, seq_len)
        alibi_bias = self.slopes[:, None, None] * relative_position[None, :, :]

        return alibi_bias

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask (torch.Tensor, optional): Optional mask of shape (batch_size, seq_len). Defaults to None.
            casual_mask (bool, optional): Whether to apply causal masking. Defaults to True.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)

        # Add ALiBi biases
        # Shape: (n_heads, seq_len, seq_len) -> broadcast to (batch_size, n_heads, seq_len, seq_len)
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        scores = scores + alibi_bias

        # Apply causal mask if needed
        if causal_mask:
            causal_mask_matrix = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            )
            scores = scores + causal_mask_matrix

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expanded mask to (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, V)

        # Reshape and apply output projection
        # Shape: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class FusedALiBiAttention(nn.Module):
    """
    Optimized ALiBi with pre-computed bias cache for common sequence lengths.
    Reduce bias computation overhead during training.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_cached_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute and cache ALiBi biases
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)

        # Cache biases for common lengths
        self.max_cached_len = max_cached_len
        self._build_bias_cache(max_cached_len)

    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Same as standard implementation"""
        def get_slopes_power_of_2(n: int) -> torch.Tensor:
            start = 2 ** (-8 / n)
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)

        closest_power = 2 ** math.floor(math.log2(n_heads))
        base_slopes = get_slopes_power_of_2(closest_power)
        extra_slopes = get_slopes_power_of_2(
            2 * closest_power)[::2][:n_heads - closest_power]
        return torch.cat([base_slopes, extra_slopes])

    def _build_bias_cache(self, max_len: int):
        """Pre-compute biases for all lengths up to max_len"""
        context_position = torch.arange(max_len)[:, None]
        memory_position = torch.arange(max_len)[None, :]
        relative_position = memory_position - context_position

        # Shape: (n_heads, max_len, max_len)
        alibi_bias = self.slopes[:, None, None] * relative_position[None, :, :]
        self.register_buffer('bias_cache', alibi_bias)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """Forward pass with cached biases and fused projections."""
        batch_size, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        # (3, batch, heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)

        # Use cached bias if available
        if seq_len <= self.max_cached_len:
            alibi_bias = self.bias_cache[:, :seq_len, :seq_len]
        else:
            # Compute on-the-fly for longer sequences
            alibi_bias = self._compute_bias(seq_len, x.device)

        scores = scores + alibi_bias

        if causal_mask:
            causal_mask_matrix = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            )
            scores = scores + causal_mask_matrix

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output

    def _compute_bias(self, seq_len: int, device):
        """Fallback for sequences longer than cache."""
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position
        return self.slopes[:, None, None] * relative_position[None, :, :]


class FlashALiBiAttention(nn.Module):
    """
    Memory-efficient ALiBi using tiled computation
    Computes attention in blocks to reduce memory footprint
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('slopes', self._get_slopes(n_heads))

    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Same slope computation as standard implementation"""
        def get_slopes_power_of_2(n: int) -> torch.Tensor:
            start = 2 ** (-8 / n)
            ratio = start
            return torch.Tensor([start * (ratio ** i) for i in range(n)])

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)

        closest_power = 2 ** math.floor(math.log2(n_heads))
        base_slopes = get_slopes_power_of_2(closest_power)
        extra_slopes = get_slopes_power_of_2(
            2 * closest_power)[::2][:n_heads - closest_power]
        return torch.cat([base_slopes, extra_slopes])

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """
        Block-wise attention computation to reduce memory usage.
        Useful for very long sequences during inference.
        """
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)

        # For short sequences, use standard computation
        if seq_len <= self.block_size:
            return self._standard_forward(Q, K, V, seq_len, x.device, causal_mask)

        # Block-wise computation for long sequences
        output = torch.zeros_like(Q)

        for q_start in range(0, seq_len, self.block_size):
            q_end = min(q_start + self.block_size, seq_len)
            q_block = Q[:, :, q_start:q_end, :]

            # For causal attention, only attend to previous positions
            k_end = q_end if causal_mask else seq_len
            k_block = K[:, :, :k_end, :]
            v_block = V[:, :, :k_end, :]

            # Compute attention for this block
            scores = torch.matmul(
                q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Add ALiBi bias for this block
            bias = self._compute_block_size(q_start, q_end, k_end, x.device)
            scores = scores + bias

            if causal_mask:
                # Causal mask within block
                mask = torch.triu(
                    torch.ones(q_end - q_start, k_end,
                               device=x.device) * float('-inf'),
                    diagonal=k_end - q_end + 1
                )
                scores = scores + mask

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output[:, :, q_start:q_end, :] = torch.matmul(
                attn_weights, v_block)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        return self.out_proj(output)

    def _compute_block_bias(self, q_start: int, q_end: int, k_end: int, device):
        """Compute ALiBi bias for a specific query-key block."""
        context_position = torch.arange(q_start, q_end, device=device)[:, None]
        memory_position = torch.arange(k_end, device=device)[None, :]
        relative_position = memory_position - context_position
        return self.slopes[:, None, None] * relative_position[None, :, :]

    def _standard_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        seq_len: int,
        causal_mask: bool,
        device,
    ) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)

        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position
        alibi_bias = self.slopes[:, None, None] * relative_position[None, :, :]
        scores = scores + alibi_bias

        if causal_mask:
            causal_mask_matrix = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            scores = scores + causal_mask_matrix

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        batch_size = Q.shape[0]
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        return output


if __name__ == "__main__":
    # Initialize ALiBi attention layer
    d_model = 512
    n_heads = 8
    batch_size = 16
    seq_len = 1024

    attention = ALiBiAttention(d_model=d_model, n_heads=n_heads)

    # Input: (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass with causal masking (for language modeling)
    output = attention(x, causal_mask=True)
    # Output shape: (16, 1024, 512)

    # Extrapolation: inference on longer sequences
    seq_len_inference = 2048
    x_long = torch.randn(batch_size, seq_len_inference, d_model)
    output_long = attention(x_long, causal_mask=True)
    # Output shape: (16, 2048, 512) - extrapolates beyond training length
