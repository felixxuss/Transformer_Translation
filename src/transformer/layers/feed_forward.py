from torch import nn

class FeedForward(nn.Module):
    def __init__(self, input_dim, feature_dim) -> None:
        """Position wise feed forward layer.

        Args:
            input_dim: Embedding dimension of the input. 
            feature_dim: Hidden dimension of the position wise feed forward layer.
        """
        # Practical 7: feature_dim is the hidden dimension of the position wise feed forward layer
        super().__init__()

        self.linear1 = nn.Linear(input_dim, feature_dim) # was n_embed, n_embed*4
        self.linear2 = nn.Linear(feature_dim, input_dim) # was n_embed*4, n_embed

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the feed forward layer.

        Args:
            x: Tensor of shape (batch_size, context_length, embedding_size).
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x