import torch
import torch.nn as nn
import math


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention with shared key-value heads.

    Args:
        embed_dim (int): Total embedding dimension (C)
        num_heads (int): Number of query heads (H)
        dropout (float): Dropout probability (Default = 0.1)
        bias (bool): Whether to use bias in projections (Default = True)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query projection: full dimensionality
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Key and value projections: single head only
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None
    ):
        """
        Forward pass with optional KV caching.

        Args:
            x (torch.Tensor): Input Tensor [B, T, C]
            attn_mask (torch.Tensor, optional): Optional attention mask [B, 1, T, T] or [T, T]. Defaults to None.
            kv_cache (tuple[torch.Tensor, torch.Tensor], optional): Optional tuple of (cached_k, cached_v) for inference. Defaults to None.

        Returns:
            output: [B, T, C]
            new_kv_cache: Updated (k, v) cache if kv_cache was provided.
        """
        B, T, C = x.shape

        # Project queries: [B, T, C] -> [B, H, T, D_h]
        q = self.q_proj(x).view(B, T, self.num_heads,
                                self.head_dim).transpose(1, 2)

        # Project keys and values: [B, T, C] -> [B, 1, T, D_h]
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        # Handle KV cache for inference
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # Append along sequence dimension
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        # Compute attention scores: [B, H, T, T]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [B, H, T, D_h]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output: [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)

        if new_kv_cache is not None:
            return output, new_kv_cache
        return output


class FusedMQA(nn.Module):
    """MQA using PyTorch's fused scaled_dot_product_attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        # Expand k and v to match q's head dimension
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        # Use fused kernel
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


class PagedMQA(nn.Module):
    """MQA with paged KV cache for efficient memory management"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 16,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def allocate_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device,
        dtype=torch.float16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-allocate paged KV cache."""
        num_blocks = (max_seq_len + self.block_size - 1) // self.block_size
        k_cache = torch.zeros(
            max_batch_size, num_blocks, self.block_size, self.head_dim,
            device=device, dtype=dtype
        )
        v_cache = torch.zeros(
            max_batch_size, num_blocks, self.block_size, self.head_dim,
            device=device, dtype=dtype
        )
        return k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        cache_positions: torch.Tensor = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, self.head_dim)
        v = self.v_proj(x).view(B, T, 1, self.head_dim)

        if kv_cache is not None and cache_positions is not None:
            k_cache, v_cache = kv_cache
            # Write to paged cache at specified positions
            for i, pos in enumerate(cache_positions):
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                k_cache[i, block_idx, block_offset] = k[i, 0, 0]
                v_cache[i, block_idx, block_offset] = v[i, 0, 0]

            # Read full cache for attention
            k = k_cache.view(B, -1, 1, self.head_dim).transpose(1, 2)
            v = v_cache.view(B, -1, 1, self.head_dim).transpose(1, 2)
        else:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


if __name__ == "__main__":
    # Initialize MQA layer
    embed_dim = 512
    num_heads = 8
    mqa = MultiQueryAttention(embed_dim, num_heads, dropout=0.1)

    # Training forward pass
    batch_size = 4
    seq_len = 128
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Without caching (training)
    output = mqa(x)  # Shape: [4, 128, 512]

    # With causal masking
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf')),
        diagonal=1
    )
    output = mqa(x, attn_mask=causal_mask)
