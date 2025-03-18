import torch
from torch import nn
from layer_norm import LayerNorm
from dummy_gpt_model import DummyGPTModel

# 4.1 Coding an LLM architecture

"""
we have done:
* tokenization
* positional encoding
* multi-head attention

we are going to scale up to GPT-2 size model, with 124 million parameters
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print(f"Output shape: {logits.shape}")
print(logits)

# 4.2 Normalizing activations with layer normalization

"""
Training neural networks with many layers can sometimes have issues like vanishing or exploding gradients. This means that the gradients (the values used to update the model's parameters) can become very small or very large, making it hard for the model to learn. To help with this, we can use a technique called layer normalization.

Layer noramlization improves stability and efficiency of training. The idea is to adjust the activations (outputs) of a neural network to have a mean of 0 and a variance of 1. This makes convergence faster and more stable. It also helps the model learn better by making the training process smoother.

In a GPT-like network, this usually happens before and after the multi-head attention and feed-forward layers. The layer normalization is applied to the input of the multi-head attention and the output of the feed-forward layer. This helps to keep the activations in a good range, making it easier for the model to learn.
"""

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)  # mean of the last dimension
var = out.var(dim=-1, keepdim=True)
print(f"Mean:\n{mean}")
print(f"Variance:\n{var}")

# apply layer normalization

torch.set_printoptions(sci_mode=False)
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print(f"Normalized layer outputs:\n{out_norm}")
print(f"Mean after normalization:\n{mean}")
print(f"Variance after normalization:\n{var}")

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print(f"Mean after layer normalization:\n{mean}")
print(f"Variance after layer normalization:\n{var}")
