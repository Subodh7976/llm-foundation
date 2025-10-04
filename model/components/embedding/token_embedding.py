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
