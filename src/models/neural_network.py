"""
PyTorch Neural Network Architecture for Math Olympiad Problem Classification
=============================================================================

Architecture
------------
``MathProblemClassifier`` — an embedding-based feed-forward network:

  Token IDs → Embedding → Mean-Pool → FC Stack → Softmax

The embedding layer converts sparse integer token IDs into dense vectors.
Mean-pooling over the sequence dimension produces a fixed-size representation
regardless of sequence length, making this architecture simple yet effective
for multi-class text classification.

``MathProblemDataset`` — a ``torch.utils.data.Dataset`` wrapper that pairs
pre-computed integer sequences with integer class labels.
"""

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MathProblemDataset(Dataset):
    """
    PyTorch Dataset wrapping token-index sequences and their labels.

    Parameters
    ----------
    sequences : list[list[int]]
        Each inner list is a fixed-length padded token-index sequence.
    labels : list[int]
        Integer class labels aligned with *sequences*.
    """

    def __init__(self, sequences: List[List[int]], labels: List[int]) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MathProblemClassifier(nn.Module):
    """
    Embedding + Mean-Pool + Fully-Connected classifier.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens (including PAD and UNK).
    embedding_dim : int
        Dimensionality of the token embedding vectors.
    hidden_dims : list[int]
        Sizes of hidden fully-connected layers applied after pooling.
    num_classes : int
        Number of output categories.
    dropout : float
        Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Embedding layer (index 0 = PAD → zeroed out via padding_idx) ──
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # ── Fully-connected stack ─────────────────────────────────────────
        layers: List[nn.Module] = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.fc_stack = nn.Sequential(*layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(batch_size, seq_len)`` — padded token-index sequences.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch_size, num_classes)``.
        """
        # (B, L) → (B, L, E)
        embedded = self.embedding(token_ids)

        # Mean-pool over the sequence dimension: (B, L, E) → (B, E)
        # Padding positions carry zero vectors (padding_idx=0) so mean-pooling
        # is unaffected only when the whole sequence is PAD, which is excluded
        # by the DataIngestion min_text_length guard.
        pooled = embedded.mean(dim=1)

        # (B, E) → (B, num_classes)
        return self.fc_stack(pooled)

    def predict_proba(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities (softmax over logits).

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, num_classes)``.
        """
        with torch.no_grad():
            logits = self.forward(token_ids)
            return torch.softmax(logits, dim=-1)
