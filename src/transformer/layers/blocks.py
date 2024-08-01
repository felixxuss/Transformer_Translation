from torch import nn
from src.transformer.layers.attention import MultiHeadAttention
from src.transformer.layers.feed_forward import FeedForward

class BaseTransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.2) -> None:
        """Base transformer layer. Contains self attention and feed forward layer. Used in encoder.

        Args:
            input_dim: Embedding dimension of the input. 
            num_heads: Number of heads in the multi head attention layer.
            feature_dim: Hidden dimension of the position wise feed forward layer.
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False) # no future masking in the encoder
        self.feature_transformation = FeedForward(input_dim, feature_dim)
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, attention_mask):
        """Forward pass through the encoder layer.

        Args:
            x: Tensor of shape (batch_size, context_length, embedding_size).
            attention_mask: Mask for the attention.
        """
        # self attention
        y = self.self_attention(x, x, x, attention_mask)
        y *= attention_mask.unsqueeze(-1).float() # Why?

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_1(x + y)

        y = self.feature_transformation(x)
        y *= attention_mask.unsqueeze(-1).float() # Why?

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_2(x + y)

        x *= attention_mask.unsqueeze(-1).float() # Why?

        return x
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.2):
        """Transformer decoder layer. Contains self attention, cross attention and feed forward layer. Used in decoder.

        Args:
            input_dim: Embedding dimension of the input.
            num_heads: Number of heads in the multi head attention layer.
            feature_dim: Hidden dimension of the position wise feed forward layer.
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True) # future masking in the decoder
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)
        self.feature_transformation = FeedForward(input_dim, feature_dim)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, encoder, encoder_attention_mask, attention_mask):
        """Forward pass through the decoder layer.

        Args:
            x: Tensor of shape (batch_size, context_length, embedding_size).
            encoder: Value tensor from the encoder. Tensor of shape (batch_size, context_length, embedding_size).
            encoder_attention_mask: Mask for the encoder attention.
            attention_mask: Mask for the decoder attention.
        """
        y = self.self_attention(x, x, x, attention_mask)
        y *= attention_mask.unsqueeze(-1).float() # Why?

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_1(x + y)

        # in cross attention, the query comes from the decoder and the key and value come from the encoder
        y = self.encoder_attention(x, encoder, encoder, encoder_attention_mask)
        y *= attention_mask.unsqueeze(-1).float() # Why?

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_2(x + y)

        y = self.feature_transformation(x)
        y *= attention_mask.unsqueeze(-1).float() # Why?

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_3(x + y)
        x *= attention_mask.unsqueeze(-1).float() # Why?

        return x