import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, d_model, n_vocab=50_000):
        super(TokenEmbedding, self).__init__()
        self.tok_embed = nn.Embedding(n_vocab, d_model)

    def forward(self, x):
        # x.shape = (batch_size, context_length)
        out = self.tok_embed(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=64):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        """returns the positional embeddings of an input x
    
        embedding shape: (batch_size, context_length, d_model)

        Args:
            x (_type_): tokenized input with shape (batch_size, context_length, embed_size)
        """  
        B, seq_len = x.shape[0], x.shape[1]
        position = torch.arange(0, seq_len).unsqueeze(-1)
        pos_enc = torch.zeros((seq_len, self.d_model))

        div_term = 1 / torch.pow(10000, 2 * (torch.arange(0, self.d_model//2) / self.d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        batch_pos_enc = torch.tile(pos_enc, (B, 1, 1))
        return batch_pos_enc


