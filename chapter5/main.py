# 5.1 Evaluating generative text models
import torch
import tiktoken

from chapter4.gpt_model import GPTModel
from chapter4.util import generate_text_simple
from .util import text_to_token_ids, token_ids_to_text

"""
we will set up our model for text generation and look at basic ways to evaluate the quality of the generated text, and calculate the training and validation losses
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_content = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_content, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print(f"Output text: {token_ids_to_text(token_ids, tokenizer)}")
