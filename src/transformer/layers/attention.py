import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, mask_future=True) -> None:
        """Attention layer.

        Args:
            mask_future (bool, optional): Defaults to True.
        """
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask):
        """Forward pass through the attention layer.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim).
            key: Key tensor of shape (batch_size, seq_len, embed_dim).
            value: Value tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Mask for the attention.

        Returns:
            _type_: _description_
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention = query @ key.transpose(-1, -2)

        # key.shape[-1] = embed dimension
        attention /= (key.shape[-1] ** 0.5) 

        # mask future for the decoder component
        if self.mask_future:
            # query.shape[1] = len of input, key.shape[1] = len of output
            forward_mask = torch.tril(torch.ones(query.shape[1], key.shape[1])).to(device)
            attention = attention.masked_fill(forward_mask == 0, -torch.inf)

        # mask padding
        # padding was introduced to make all sentences the same length
        attention_mask = attention_mask.unsqueeze(1)
        attention = attention.masked_fill(attention_mask == 0, -torch.inf)

        attention = F.softmax(attention, dim=-1)
        
        attention = attention @ value

        return attention
 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False, dropout=0.2) -> None:
        """Multi head attention layer.

        Args:
            d_model: Embedding dimension.
            n_heads: Number of attention heads.
            mask_future (bool, optional): Defaults to False.
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.dk = d_model // n_heads # dimension of each attention head
        
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)

        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.self_attention = Attention(mask_future=mask_future)
    
    def forward(self, x_query, x_key, x_value, attention_mask):
        # apply linear transformations
        Q = self.query_transform(x_query) # (batch_size, seq_len, d_model)
        K = self.key_transform(x_key) # (batch_size, seq_len, d_model)
        V = self.value_transform(x_value) # (batch_size, seq_len, d_model)

        # split into heads
        Qs = Q.split(self.dk, dim=-1) # (batch_size, seq_len, dk)
        Ks = K.split(self.dk, dim=-1) # (batch_size, seq_len, dk)
        Vs = V.split(self.dk, dim=-1) # (batch_size, seq_len, dk)

        # pass through attention
        x = []
        for q, k, v in zip(Qs, Ks, Vs):
            x.append(self.self_attention(q, k, v, attention_mask))

        # concat heads
        x_concat = torch.cat(x, dim=-1)

        # apply output transformation
        x = self.output_transform(x_concat)

        return x