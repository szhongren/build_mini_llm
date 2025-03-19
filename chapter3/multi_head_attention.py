import torch
from torch import nn
from .causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # print(f"keys: {keys}")
        # print(f"keys.shape: {keys.shape}")

        # split embeddings into multiple heads, allow each head to focus on different parts of the input
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        # print(f"keys: {keys}")
        # print(f"keys.shape: {keys.shape}")
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # rearrange tensor dimensions to allow for batch matrix multiplication, more efficient
        keys = keys.transpose(1, 2)  # fix the key dimension reshaping
        # print(f"keys: {keys}")
        # print(f"keys.shape: {keys.shape}")
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        #
        attention_scores = queries @ keys.transpose(2, 3)
        # print(f"keys.transpose(2, 3): {keys.transpose(2, 3)}")
        # print(f"keys.transpose(2, 3).shape: {keys.transpose(2, 3).shape}")
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # transpose back to shape of (b, num_tokens, num_heads, head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2)

        # flatten to shape of (b, num_tokens, d_out)
        # combine the output of all heads
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

        # add an output projection, this is not strictly necessary but common
        context_vector = self.out_proj(context_vector)
        return context_vector
