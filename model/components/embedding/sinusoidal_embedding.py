import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding from "Attention is All you Need"

    Args:
        d_model (int): dimensions of embedding (must be even)
        max_len (int): Maximum sequence length to pre-compute
        dropout (float): Dropout rate applied to positional encodings
        base (int): Base for geometric progression (default: 10000)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        base: int = 10000
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Pre-compute positional encodings
        pe = self._compute_positional_encodings(max_len, d_model, base)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe, persistent=False)

    def _compute_positional_encodings(
        self,
        max_len: int,
        d_model: int,
        base: int
    ) -> torch.Tensor:
        """
        Compute Sinusoidal positional encodings

        Returns:
            torch.Tensor: Tensor of shape (max_len, d_model)
        """
        # Create position indices [0, 1, 2 ..., max_len - 1]
        position = torch.arange(max_len).unsqueeze(1).float()

        # Create dimension indices [0, 1, 2, ... d_model - 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(base) / d_model)
        )

        # Initialize encoding matrix
        pe = torch.zeros(max_len, d_model)

        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor = None,
        seq_length: int = None,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor, optional): input tensor of shape (B, T, C) or (T, C). Defaults to None.
            seq_length (int, optional): If x not provided, length of sequence. Defaults to None.
            position_ids (torch.Tensor, optional): Optional specific positions of shape (T) or (B, T). Defaults to None.

        Returns:
            torch.Tensor: Positional encodings of shape (T, C) or (B, T, C)
        """
        if x is not None:
            seq_length = x.size(-2)   # Works for both (B, T, C) or (T, C)

        if seq_length is None and position_ids is None:
            raise ValueError("Must provide x, seq_length, or position_ids")

        if position_ids is not None:
            # Use specific positions
            if position_ids.max() >= self.max_len:
                raise ValueError(
                    f"Position {position_ids.max()} exceeds max_len {self.max_len}"
                )
            pos_encoding = self.pe[position_ids]
        else:
            # Use sequential positions
            if seq_length > self.max_len:
                raise ValueError(
                    f"Sequence length {seq_length} exceeds max_len {self.max_len}"
                )
            pos_encoding = self.pe[:seq_length]

        # Apply dropout
        pos_encoding = self.dropout(pos_encoding)
        return pos_encoding


class SinusoidalPositionalEncodingVectorized(nn.Module):
    """
    Optimized sinusoidal positional encoding with fully vectorized computation
    Faster pre-computation using broadcasting
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
        base: int = 10000
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Pre-compute using vectorized operations
        pe = self._compute_vectorized(max_len, d_model, base)
        self.register_buffer('pe', pe, persistent=False)

    def _compute_vectorized(
        self,
        max_len: int,
        d_model: int,
        base: int
    ) -> torch.Tensor:
        """
        Fully vectorized computation without loops
        """
        # Positional indices: shape (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # Frequency for each dimension pair: shape (d_model//2,)
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        frequencies = 1.0 / (base ** (dim_indices / d_model))

        # Compute angles: shape (max_len, d_model//2)
        angles = position * frequencies.unsqueeze(0)

        # Stack sine and cosine: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe

    def forward(
        self,
        x: torch.Tensor = None,
        seq_length: int = None,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        if x is not None:
            seq_length = x.size(-2)

        if position_ids is not None:
            return self.dropout(self.pe[position_ids])

        if seq_length is None:
            raise ValueError("Must provide x, seq_length or position_ids")

        return self.dropout[self.pe[:seq_length]]


class SinusoidalPositionalEncodingExtended(nn.Module):
    """
    Sinusoidal encoding with dynamic extension for longer sequences
    Computes encodings on-the-fly for positions beyond pre-computed cache
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
        base: int = 10000,
        extend_factor: float = 1.0
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be an integer, got {d_model}")

        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.extend_factor = extend_factor

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Pre-compute standard encodings
        pe = self._compute_positional_encoding(max_len, d_model, base)
        self.register_buffer('pe', pe, persistent=False)

        # Store frequency computation for extension
        self.register_buffer(
            "frequencies",
            self._compute_frequencies(d_model, base),
            persistent=False
        )

    def _compute_frequencies(self, d_model: int, base: int) -> torch.Tensor:
        """Pre-compute frequencies for all dimensions"""
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        return 1.0 / (base ** (dim_indices / d_model))

    def _compute_positional_encoding(
        self,
        max_len: int,
        d_model: int,
        base: int
    ) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = self._compute_frequencies(d_model, base)
        angles = position * frequencies.unsqueeze(0)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe

    def _extend_encodings(self, target_len: int) -> torch.Tensor:
        """Compute encodings for longer sequences on-the-fly"""
        position = torch.arange(
            target_len,
            dtype=torch.float32,
            device=self.frequencies.device
        ).unsqueeze(1)

        # Apply extension factor for interpolation
        angles = (position * self.extend_factor) * \
            self.frequencies.unsqueeze(0)

        pe_extended = torch.zeros(
            target_len,
            self.d_model,
            device=self.frequencies.device
        )
        pe_extended[:, 0::2] = torch.sin(angles)
        pe_extended[:, 1::2] = torch.cos(angles)

        return pe_extended

    def forward(
        self,
        x: torch.Tensor = None,
        seq_length: int = None,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        if x is not None:
            seq_length = x.size(-2)

        if position_ids is not None:
            max_pos = position_ids.max().item()
            if max_pos >= self.max_len:
                # Extend encodings dynamically
                pe_extended = self._extend_encodings(max_pos + 1)
                return self.dropout(pe_extended[position_ids])
            return self.dropout(self.pe[position_ids])

        if seq_length is None:
            raise ValueError("Must provide x, seq_length, or position_ids")

        if seq_length > self.max_len:
            # Compute extended encodings
            pe_extended = self._extend_encodings(seq_length)
            return self.dropout(pe_extended)

        return self.dropout(self.pe[:seq_length])


if __name__ == "__main__":
    # Configuration
    d_model = 512
    max_len = 5000
    batch_size = 16
    seq_length = 128

    # Example 1: Standard sinusoidal encoding
    sin_pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=1.0)

    # Get encodings for a sequence
    pos_enc = sin_pos(seq_length=seq_length)
    print(f"Positional encoding shape: {pos_enc.shape}")  # (128, 512)

    # Example 2: With input tensor
    x = torch.randn(batch_size, seq_length, d_model)
    pos_enc = sin_pos(x=x)
    print(f"From input tensor: {pos_enc.shape}")  # (128, 512)

    # Example 3: Specific positions
    position_ids = torch.tensor([0, 10, 20, 30, 40])
    pos_enc = sin_pos(position_ids=position_ids)
    print(f"Specific positions: {pos_enc.shape}")  # (5, 512)

    # Example 4: Extended context
    extended_pos = SinusoidalPositionalEncodingExtended(
        d_model=d_model,
        max_len=max_len,
        extend_factor=1.0
    )
    # Can handle sequences longer than max_len
    long_pos_enc = extended_pos(seq_length=8000)
    print(f"Extended encoding: {long_pos_enc.shape}")  # (8000, 512)
