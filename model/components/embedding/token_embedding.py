import torch
import torch.nn as nn
import torch.nn.functional as f
import math


class TokenEmbedding(nn.Module):
    """
    Standard Token Embedding Layer

    Args:
        vocab_size (int): Size of vocabulary (V)
        d_model (int): Embedding dimension (C)
        padding_idx (int, optional): If specified, entries at padding_idx don't contribute to gradient
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Embedding lookup table
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot uniform initialization"""
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / math.sqrt(self.d_model),
            1.0 / math.sqrt(self.d_model)
        )

        # Zero out padding embedding if specified
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids (torch.Tensor): Integer tensor of shape (B, T)

        Returns:
            torch.Tensor: Tensor of shape (B, T, C)
        """
        return self.embedding(token_ids)


class TokenEmbeddingXavierNormal(TokenEmbedding):
    """
    Token Embedding with Xavier Normal Initialization
    Often provides better convergence then uniform
    """

    def _init_weights(self):
        """Xavier/Glorot Normal Initialization"""
        nn.init.xavier_normal_(self.embedding.weight)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)


class TokenEmbeddingMaxNorm(nn.Module):
    """
    Token Embedding with max norm constraint
    Prevents embedding vectors from growing too large
    Useful for regularization and preventing overfitting
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_norm: float = 1.0,
        padding_idx: int = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_norm = max_norm
        self.padding_idx = padding_idx

        # Embedding with max norm constraint
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            max_norm=max_norm,  # Renormalize if exceeds this value
            norm_type=2.0       # L2 Norm
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small random values"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Max norm is applied automatically during lookup
        return self.embedding(token_ids)


class TokenEmbeddingSparse(nn.Module):
    """
    Sparse Token Embedding for memory-efficient training
    Uses sparse gradients - beneficial for very large vocabularies
    Only non-zero gradient entries are stored
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Enable sparse gradients
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            sparse=True
        )

        self._init_weights()

    def _init_weights(self):
        """Normal initialization scaled by embedding dimension"""
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.d_model)
        )

        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class TokenEmbeddingScaled(nn.Module):
    """
    Token Embedding with scaling factor
    Common in Transformer models (e.g., original Transformer, BERT)
    Scales embeddings by sqrt(d_model) to balance the positional encodings
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Scaling Factor
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        self._init_weights()
    
    def _init_weights(self):
        """Normal Initialization"""
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02
        )
        
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        
        return embeddings * self.scale


class TokenEmbeddingWithTying(nn.Module):
    """
    Token Embedding designed for weight tying with output projection
    Shares weights between embedding and final linear layer (lm_head)
    Reduces parameters and often improves performance
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = None,
        scale_embeddings: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.scale_embeddings = scale_embeddings
        self.scale = math.sqrt(d_model) if scale_embeddings else 1.0
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize Weight tying"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        if self.scale_embeddings:
            embeddings = embeddings * self.scale
        return embeddings

    def get_weight_for_tying(self):
        """Returns weight matrix for tying with output layer"""
        return self.embedding.weight


if __name__ == "__main__":
    # Configuration
    vocab_size = 50000
    d_model = 768
    seq_len = 512
    batch_size = 32
    padding_idx = 0

    # Create sample token IDs
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    token_ids[:, -10:] = padding_idx  # Add some padding tokens

    # Initialize different implementations
    standard_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
    xavier_emb = TokenEmbeddingXavierNormal(vocab_size, d_model, padding_idx)
    maxnorm_emb = TokenEmbeddingMaxNorm(vocab_size, d_model, max_norm=1.0, padding_idx=padding_idx)
    sparse_emb = TokenEmbeddingSparse(vocab_size, d_model, padding_idx)
    scaled_emb = TokenEmbeddingScaled(vocab_size, d_model, padding_idx)

    # Forward pass
    embeddings = standard_emb(token_ids)

    print(f"Input shape (token IDs): {token_ids.shape}")
    print(f"Output shape (embeddings): {embeddings.shape}")
    print(f"Embedding matrix shape: {standard_emb.embedding.weight.shape}")
    print(f"Number of parameters: {vocab_size * d_model:,}")
    print(f"\nPadding embedding norm: {standard_emb.embedding.weight[padding_idx].norm():.6f}")
    print(f"Regular token embedding norm: {standard_emb.embedding.weight[100].norm():.6f}")

