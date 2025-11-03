import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention from "Attention is All You Need".

    Args:
        d_model (int): Model dimension (C)
        num_heads (int): Number of attention heads (H)
        dropout (float): Dropout probability
        bias (bool): Wether to use bias in projection
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Independent projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        kv_cache: tuple = None,
        use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            mask (torch.Tensor, optional): Optional attention mask of shape (B, 1, T, T) or (B, H, T, T). Defaults to None.
            kv_cache (tuple, optional): Optional tuple (cached_k, cached_v) for autoregressive decoding. Defaults to None.
            use_cache (bool, optional): Whether to return updated KV cache. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
            Optional: Updated kv_cache if use_cache=True
        """
        B, T, C = x.shape

        # Project to Q, K, V: (B, T, C) -> (B, T, H*Dh)
        Q = self.q_proj(x)
        K = self.q_proj(x)
        V = self.v_proj(x)

        # Reshape to separate heads: (B, T, H*Dh) -> (B, H, T, Dh)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # Concatenate along sequence dimension
            K = torch.cat([cached_k, K], dim=2)
            V = torch.cat([cached_v, V], dim=2)

        # Compute attention scores: (B, H, T, Dh) @ (B, H, Dh, T_k) -> (B, H, T, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (B, H, T, T_k) @ (B, H, T_k, Dh) -> (B, H, T, Dh)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: (B, H, T, Dh) -> (B, T, H*Dh)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        output = self.o_proj(attn_output)

        if use_cache:
            return output, (K, V)
        return output


class FusedMultiHeadAttention(nn.Module):
    """
    MHA with fused QKV projection for reduced kernel launches.
    Single matrix multiplication instead of three separate ones.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Fused QKV projection: C -> 3*C
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Single projection: (B, T, C) -> (B, T, 3*C)
        qkv = self.qkv_proj(x)

        # Split and reshape: (B, T, 3*C) -> 3 x (B, H, T, Dh)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


class FlashMultiHeadAttention(nn.Module):
    """
    MHA using Flash Attention for memory-efficient computation.
    Reduces memory from O(T^2) to O(T) by fusing operations and tiling.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Use PyTorch's scaled_dot_product_attention (Flash Attention backend)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


class CachedMultiHeadAttention(nn.Module):
    """
    MHA optimized for autoregressive generation with efficient KV caching. 
    Stores and reuses previous key-value pairs to avoid recomputation.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.max_seq_len = max_seq_len
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-allocate cache buffers
        self.register_buffer('k_cache', torch.zeros(1, num_heads, max_seq_len, self.head_dim))
        self.register_buffer('v_cache', torch.zeros(1, num_heads, max_seq_len, self.head_dim))
        self.register_buffer('cache_position', torch.tensor(0, dtype=torch.long))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)
    
    def reset_cache(self):
        self.cache_position.zero_()
    
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            # Update cache
            pos = self.cache_position.item()
            self.k_cache[:, :, pos:pos+T, :] = K
            self.v_cache[:, :, pos:pos+T, :] = V
            self.cache_position += T
            
            # Use full cache sequence
            K = self.k_cache[:, :, :pos+T, :]
            V = self.v_cache[:, :, :pos+T, :]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


if __name__ == "__main__":
    # Initialize MHA
    d_model = 512
    num_heads = 8
    batch_size = 32
    seq_len = 128

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)

    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)  # (32, 128, 512)

    # Forward pass
    output = mha(x)  # (32, 128, 512)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Create causal mask for autoregressive generation
    def create_causal_mask(seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    # Apply masked attention
    seq_len = 128
    x = torch.randn(8, seq_len, 512)
    mask = create_causal_mask(seq_len, x.device)

    mha = MultiHeadAttention(d_model=512, num_heads=8)
    output = mha(x, mask=mask)  # Only attends to previous positions

    # Prefill phase
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    prompt = torch.randn(1, 50, 512)  # Initial prompt

    # Process prompt
    output, kv_cache = mha(prompt, use_cache=True)

    # Generation loop (one token at a time)
    for _ in range(100):  # Generate 100 tokens
        new_token = torch.randn(1, 1, 512)  # New token embedding
        output, kv_cache = mha(new_token, kv_cache=kv_cache, use_cache=True)
        # output shape: (1, 1, 512) - only for new token
