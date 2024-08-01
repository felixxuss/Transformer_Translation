import torch.nn as nn

from .layers.embedding import TokenEmbedding, PositionalEncoding
from .layers.blocks import BaseTransformerBlock, TransformerDecoderBlock

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int,
                 weight_decay: float) -> None:
        """Transformer model.

        Args:
            vocab_size (int): The number of distinct tokens the model will see. Is set by the tokenizer.
            
            d_model (int): The dimensionality of the embedding layer.
            
            n_heads (int): The number of heads in the multi-head attention layers.

            num_encoder_layers (int): The number of encoder layers.
            
            num_decoder_layers (int): The number of decoder layers.
            
            dim_feedforward (int): The hidden dimension of the position wise feed forward layer.
            
            dropout (float): droput rate.
            
            max_len (int): The maximum length of the input sequence.
            
            weight_decay (float): Weight decay for the AdamW optimizer.
        """
        super(Transformer, self).__init__()


        # Embedding layers
        self.token_embedding = TokenEmbedding(d_model, vocab_size)
        self.token_embedding.tok_embed.weight.data *= d_model ** 0.5 # right scaling?

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder layers
        self.encoder = nn.ModuleList([
            BaseTransformerBlock(d_model, n_heads, dim_feedforward, dropout) 
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, dim_feedforward, dropout) 
            for _ in range(num_decoder_layers)
        ])

        # Post transfomration layers
        self.pre_sm_lin_transform = nn.Linear(d_model, vocab_size, bias=False)
        # self.softmax = nn.Softmax(dim=-1)

        # Define parameters for different groups
        params_no_weight_decay = [p for name, p in self.named_parameters() if 'bias' in name or 'layer_norm' in name]
        params_with_weight_decay = [p for name, p in self.named_parameters() if not 'bias' in name and not 'layer_norm' in name]

        # Define parameter groups with different weight decay values
        self.param_groups = [
            {'params': params_with_weight_decay, 'weight_decay': weight_decay},
            {'params': params_no_weight_decay, 'weight_decay': 0.0},  # Bias and layer normalization parameters
        ]

    def forward(self, en_batch, de_batch, de_mask, en_mask, config):
        """Currently assumes that we translate from english to german"""
        # encoder
        encoder_out = self.encode(en_batch, en_mask, config)

        # decoder
        decoder_out = self.decode(de_batch, de_mask, encoder_out, en_mask, config)

        # post transformation
        y = self.head(decoder_out)

        return y
    
    def encode(self, en_batch, en_mask, config):
        """Encodes the english sentence"""
        # embedd the tokens
        en_embed = self.token_embedding(en_batch) 
        pos_embedding = self.positional_encoding(en_embed).to(config["DEVICE"])
        en_embed += pos_embedding

        # encoder
        encoder_out = en_embed
        for encoder_layer in self.encoder:
            encoder_out = encoder_layer(encoder_out, attention_mask=en_mask)
        
        return encoder_out
    
    def decode(self, de_batch, de_mask, encoder_out, en_mask, config):
        """Decodes the german sentence"""
        # embedd the tokens
        de_embed = self.token_embedding(de_batch)  
        pos_embedding = self.positional_encoding(de_embed).to(config["DEVICE"])
        de_embed += pos_embedding

        # decoder
        decoder_out = de_embed
        for decoder_layer in self.decoder:
            decoder_out = decoder_layer(decoder_out, encoder_out, encoder_attention_mask=en_mask, attention_mask=de_mask)
        
        return decoder_out
    
    def head(self, decoder_out):
        return self.pre_sm_lin_transform(decoder_out) # (batch_size, max_len, vocab_size)
    

    def parameter_count(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)