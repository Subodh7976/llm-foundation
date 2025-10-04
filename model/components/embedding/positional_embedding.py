import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnedPositionalEmbedding(nn.Module):
    """
    Unified Learned Positional Embedding with configurable parameters

    Args:
        max_context_length (int): Maximum sequence length supported
        d_model (int): Embedding dimension
        dropout (float): Dropout rate applied to positional embeddings
        init_std (float): Standard deviation for weight initialization
        init_method (str): Initialization method ('normal', 'xavier_uniform', 'xavier_normal')
        learnable (bool): Whether embeddings are trainable (allows freezing)
    """

    def __init__(
        self,
        max_context_length: int,
        d_model: int,
        dropout: float = 0.0,
        init_std: float = 0.02,
        init_method: str = 'normal',
        learnable: bool = True
    ):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model
        self.learnable = learnable

        # Positional Embedding lookup table
        self.position_embedding = nn.Embedding(
            max_context_length,
            d_model
        )

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self._init_weights(init_method, init_std)

        # Option to freeze embeddings
        if not learnable:
            self.position_embedding.weight.requires_grad = False

    def _init_weights(self, method: str, std: float):
        """Initialize position embeddings"""
        if method == "normal":
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(self.position_embedding)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(self.position_embedding)
        elif method == "scaled_normal":
            nn.init.normal_(
                self.position_embedding,
                mean=0.0,
                std=1.0 / math.sqrt(self.d_model)
            )
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def forward(
        self,
        position_ids: torch.Tensor = None,
        seq_length: int = None,
        past_length: int = 0
    ) -> torch.Tensor:
        """
        Args:
            position_ids (torch.Tensor, optional): position indices of shape (T) or (B, T). Defaults to None.
            seq_length (int, optional): If position_ids not provided, create positions [0, seq_len]. Defaults to None.
            past_length (int, optional): Offset for cached/continuing sequences. Defaults to 0.

        Returns:
            torch.Tensor: Position embeddings of shape (T, C) or (B, T, C)
        """
        if position_ids is None:
            if seq_length is None:
                raise ValueError(
                    "Either position_ids or seq_length must be provided")

            # Create position indices [past_length, past_length + seq_length]
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device=self.position_embedding.weight.device
            )

        # Clip positions to maximum content length
        position_ids = torch.clamp(
            position_ids, 0, self.max_context_length - 1)

        # Lookup position embeddings
        pos_embeddings = self.position_embedding(position_ids)

        # Apply dropout
        pos_embeddings = self.dropout(pos_embeddings)

        return pos_embeddings

    def extend_context_length(self, new_max_length: int):
        """
        Extend maximum context: length by interpolating existing embeddings
        Useful for fine-tuning on longer sequences
        """
        if new_max_length <= self.max_context_length:
            return

        old_embeddings = self.position_embedding.weight.data
        old_length = self.max_context_length

        # Create new embedding layer
        new_embedding = nn.Embedding(new_max_length, self.d_model)

        # Interpolate embeddings for extended positions
        with torch.no_grad():
            # Copy existing embeddings
            new_embedding.weight[:old_length] = old_embeddings

            # Interpolate for new positions
            for i in range(old_length, new_max_length):
                # Linear interpolation from last two positions
                alpha = (i - old_length + 1) / \
                    (new_max_length - old_length + 1)
                new_embedding.weight[i] = (
                    old_embeddings[-1] * (1 - alpha) +
                    old_embeddings[-2] * alpha
                )

        self.position_embedding = new_embedding
        self.max_context_length = new_max_length


class LearnedPositionalEmbeddingCached(nn.Module):
    """
    Optimized for inference with cached position embeddings
    Pre-computes and caches all position embeddings to avoid repeated lookups
    """
    def __init__(
        self,
        max_context_length: int,
        d_model: int,
        init_std: float = 0.02
    ):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model
        
        self.position_embedding = nn.Embedding(max_context_length, d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=init_std)
        
        # Cache for pre-computed position embeddings
        self.register_buffer(
            'cached_positions',
            None,
            persistent=False
        )
    
    def _build_cache(self, device):
        """Pre-compute all position embeddings"""
        position_ids = torch.arange(
            self.max_context_length,
            dtype=torch.long,
            device=device
        )
        self.cached_positions = self.position_embedding(position_ids)
    
    def forward(
        self,
        position_ids: torch.Tensor = None,
        seq_length: int = None,
        past_length: int = 0
    ) -> torch.Tensor:
        """
        Uses cached embeddings for faster during inference
        """
        device = self.position_embedding.weight.device
        
        # Build cache if not exists
        if self.cached_positions is None or self.cached_positions.device != device:
            self._build_cache(device)
        
        if position_ids is None:
            if seq_length is None:
                raise ValueError("Either position_ids or seq_length must be provided")

            # Slice from cached embeddings
            return self.cached_positions[past_length:past_length + seq_length]

        # Use cached embeddings for indexing
        position_ids = torch.clamp(position_ids, 0, self.max_context_length - 1)
        return self.cached_positions[position_ids]


class LearnedPositionalEmbeddingExtrapolate(nn.Module):
    """
    Position embeddings with extrapolation strategies for sequences beyond training length
    Implements multiple strategies: clipping, interpolation, and learned extrapolation
    """
    def __init__(
        self,
        max_context_length: int,
        d_model: int,
        extrapolation_method: str = "clip", # 'clip', 'interpolate', 'linear'
        init_std: float = 0.02
    ):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model
        self.extrapolation_method = extrapolation_method
        
        self.position_embedding = nn.Embedding(max_context_length, d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=init_std)
        
        # For linear extrapolation: learn a direction vector
        if extrapolation_method == "linear":
            self.extrapolation_direction = nn.Parameter(
                torch.zeros(d_model)
            )
    
    def forward(
        self,
        position_ids: torch.Tensor = None,
        seq_length: int = None,
        past_length: int = 0
    ) -> torch.Tensor:
        """
        Handles positions beyond max_context_length using specified strategy
        """
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device=self.position_embedding.weight.device
            )
        
        # Check if any positions exceed max length
        max_pos = position_ids.max().item()
        
        if max_pos < self.max_context_length:
            # All positions within trained range
            return self.position_embedding(position_ids)
        
        # Handle extrapolation
        if self.extrapolation_method == "clip":
            # Clip to maximum position
            clipped_ids = torch.clamp(position_ids, 0, self.max_context_length-1)
            return self.position_embedding(clipped_ids)
        
        elif self.extrapolation_method == "interpolate":
            # Interpolate using last embedding
            embeddings = torch.zeros(
                *position_ids.shape,
                self.d_model,
                device=position_ids.device,
                dtype=self.position_embedding.weight.dtype
            )
            
            # Use learned embeddings for in-range positions
            in_range_mask = position_ids < self.max_context_length
            if in_range_mask.any():
                embeddings[in_range_mask] = self.position_embedding(
                    position_ids[in_range_mask]
                )
            
            # Use last embedding for out-of-range positions
            if (~in_range_mask).any():
                last_embedding = self.position_embedding.weight[-1]
                embeddings[~in_range_mask] = last_embedding
            
            return embeddings
        
        elif self.extrapolation_method == "linear":
            # Linear extrapolation from last position
            embeddings = torch.zeros(
                *position_ids.shape,
                self.d_model,
                device=position_ids.device,
                dtype=self.position_embedding.weight.dtype
            )
            
            in_range_mask = position_ids < self.max_context_length
            if in_range_mask.any():
                embeddings[in_range_mask] = self.position_embedding(
                    position_ids[in_range_mask]
                )
            
            if (~in_range_mask).any():
                last_embedding = self.position_embedding.weight[-1]
                out_of_range_ids = position_ids[~in_range_mask]
                
                # Extrapolate: last_emb + (pos - max_pos) * direction
                offsets = (out_of_range_ids - self.max_context_length + 1).float()
                embeddings[~in_range_mask] = (
                    last_embedding + 
                    offsets.unsqueeze(-1) * self.extrapolation_direction
                )
            
            return embeddings
        
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation_method}")


class LearnedPositionalEmbeddingBias(nn.Module):
    """
    Learned positional bias added to attention scores (AliBi-inspired)
    Instead of adding to embeddings, adds position-dependent bias to attention
    Better extrapolation properties than absolute positional embeddings
    """
    def __init__(
        self,
        num_heads: int,
        max_context_length: int,
        learnable_slopes: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_context_length = max_context_length
        
        if learnable_slopes:
            # Learn slopes for each attention head
            self.slopes = nn.Parameter(torch.ones(num_heads))
        else:
            # Fixed geometric slopes (like ALiBi)
            slopes = torch.tensor([
                2 ** (-8 * (i + 1) / num_heads)
                for i in range(num_heads)
            ])
            self.register_buffer('slopes', slopes)
        
        # Pre-compute relative position matrix
        positions = torch.arange(max_context_length)
        relative_positions = positions[None, :] - positions[:, None]
        self.register_buffer('relative_positions', relative_positions)
    
    def forward(self, seq_length: int) -> torch.Tensor:
        """
        Returns positional bias to add to attention scores
        
        Args:
            seq_length (int): current sequence length
        
        Returns:
            torch.Tensor: Bias tensor of shape (num_heads, seq_length, seq_length)
        """
        # Got relative positions for current sequence
        rel_pos = self.relative_positions[:seq_length, :seq_length]
        
        # Apply learned sloped: bias = -slope * |relative_positions|
        bias = -torch.abs(rel_pos).unsqueeze(0) * self.slopes.view(-1, 1, 1)
        return bias


if __name__ == "__main__":
    # Configuration
    max_context_length = 2048
    d_model = 768
    batch_size = 16
    seq_length = 512

    # Initialize different implementations
    standard_pos = LearnedPositionalEmbedding(
        max_context_length=max_context_length,
        d_model=d_model,
        dropout=0.1,
        init_method='normal',
        init_std=0.02
    )

    cached_pos = LearnedPositionalEmbeddingCached(
        max_context_length=max_context_length,
        d_model=d_model
    )

    extrapolate_pos = LearnedPositionalEmbeddingExtrapolate(
        max_context_length=max_context_length,
        d_model=d_model,
        extrapolation_method='linear'
    )

    # Example 1: Standard usage with automatic position IDs
    pos_emb = standard_pos(seq_length=seq_length)
    print(f"Position embeddings shape: {pos_emb.shape}")  # (512, 768)

    # Example 2: With explicit position IDs
    position_ids = torch.arange(seq_length)
    pos_emb = standard_pos(position_ids=position_ids)
    print(f"With position IDs: {pos_emb.shape}")  # (512, 768)

    # Example 3: Batched positions
    position_ids_batched = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    pos_emb_batched = standard_pos(position_ids=position_ids_batched)
    print(f"Batched positions: {pos_emb_batched.shape}")  # (16, 512, 768)

    # Example 4: Cached inference (faster for repeated calls)
    pos_emb_cached = cached_pos(seq_length=seq_length)
    print(f"Cached embeddings: {pos_emb_cached.shape}")

    # Example 5: Extrapolation beyond trained length
    long_seq_length = 3000  # Beyond max_context_length
    pos_emb_long = extrapolate_pos(seq_length=long_seq_length)
    print(f"Extrapolated positions: {pos_emb_long.shape}")  # (3000, 768)

    # Example 6: Extending context length
    standard_pos.extend_context_length(4096)
    print(f"Extended max context: {standard_pos.max_context_length}")

