import torch
from torch import nn

# 4.1 Coding an LLM architecture

from .dummy_gpt_model import DummyGPTModel

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

from .layer_norm import LayerNorm

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

# 4.3 Implementing the feed forward network with GELU activation

"""
We will implement a small neural network submodule used as part of the transformer block

historically, ReLU activation function has been used because it's simple and effective, but in LLMs, GELU (Gaussian Error Linear Unit) activation has been found to perform better in practice due to its smoother gradient and improved convergence properties. Another one that has been used is SwiGLU (Swish-gated linear unit)

GELU is defined as GELU(x) = x⋅𝛷(x), where 𝛷(x) is the cumulative distribution function of the standard Gaussian distribution.

approximate GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
"""

import matplotlib.pyplot as plt
from .feed_forward import GELU, FeedForward


def show_graphs():
    gelu, relu = GELU(), nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

# 4.4 adding shortcut connections

from .deep_neural_network import ExampleDeepNeuralNetwork

"""
Also known as skip or residual connections, 

In neural networks, shortcut connections allow the gradient to flow through the network more easily during backpropagation. 
This helps to mitigate the vanishing gradient problem, enabling the training of deeper networks. 
Shortcut connections can be implemented by adding the input of a layer to its output, allowing the model to learn an identity function if that is optimal. Conceptually, this means that the model can learn to "skip" certain layers if they are not needed for a particular input.
"""

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print_gradients(model_without_shortcut, sample_input)

sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print_gradients(model_with_shortcut, sample_input)

# 4.5 Connecting attention and linear layers in a transformer block

from .transformer_block import TransformerBlock

"""
self attention identifies and analyzes relationships between all tokens in a sequence, allowing the model to focus on relevant parts of the input.

feed forward network modifies the data individually for each token, allowing the model to learn complex patterns and relationships.

the transformer block combines these two components, enabling the model to learn both global and local patterns in the data.
"""

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
