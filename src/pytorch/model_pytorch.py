from typing import Optional

import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    Layer Normalization - implemented as nn LayerNorm does not have bias = 0
    """
    def __init__(self, eps=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim = True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # num of dimensions
        self.vocab_size = vocab_size # vocab size in english (language to be translated)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # since we multiple attention values by sqrt of d, we multiply here -> empirical result in paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        This module learns positional embeddings for input sequences, incorporating positional information
        into the embeddings by using sinusoidal functions.
        """
        super().__init__()
        self.d_model = d_model # dimension
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a zero tensor of shape (seq_len, d_model) for positional embeddings
        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)

        # Register the positional encodings as a buffer to avoid it being considered as a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape(1), :]).requires_grad_(False) #(batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads # num of heads

        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_keys = d_model // heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask, dropout: nn.Dropout) -> torch.Tensor:
        # mask can be encoder mask or decoder mask
        d_k = queries.shape[-1]

        #  transpose the last two dimensions of the keys tensor, which switches the rows and columns.
        #  computes the dot product and scales the attention scores
        attention_scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # used in decoder
            attention_scores.masked_fill_(mask == 0, -1e9)
            # fill with a small number wherever mask = 0
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ values), attention_scores # (batch, heads, seq_len)  -> (batch, heads, seq_len, d_k)


    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None):
        queries = self.w_q(queries) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        keys = self.w_k(keys) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        values = self.w_v(values) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
        queries = queries.view(queries.shape[0], queries.shape[1], self.heads, self.d_keys).transpose(1, 2)
        keys = keys.view(keys.shape[0], keys.shape[1], self.heads, self.d_keys).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.heads, self.d_keys).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(queries, keys, values, mask, self.dropout)

        # Combine heads together
        # (batch, heads, seq_len, d_k) ->  (batch, seq_len, heads, d_k) -> (batch, seq_len, h*d_k or d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_keys)
        # continuous memory for the data
        return self.w_out(x)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask) -> torch.Tensor:
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, cross_attention_block, feed_forward_block, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output, src_mask, tgt_mask) -> torch.Tensor:
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)


    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N:int = 6, h:int = 8, dropout: float = 0.1, d__ff: int = 2048):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d__ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d__ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer



