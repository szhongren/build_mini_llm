# Import necessary modules for downloading GPT-2 weights and creating model
from chapter4.gpt_model import GPTModel
from .gpt_download import download_and_load_gpt2
from .main import (
    GPT_CONFIG_124M,
    generate,
    token_ids_to_text,
    text_to_token_ids,
    tokenizer,
)
import torch

# Download and load pre-trained GPT-2 124M model parameters from OpenAI
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print(f"Settings: {settings}")
print(f"Parameter dictionary keys: {params.keys()}")
print(params["wte"])
print(f"Token embedding weight tensor dimensions:{params['wte'].shape}")

# Configuration dictionary for different GPT-2 model sizes
# Each model has different embedding dimensions, layer counts, and attention heads
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Select which model configuration to use
model_name = "gpt2-small (124M)"
# Create new configuration by copying base config and updating with model-specific settings
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"Context_length": 1024})  # Set maximum sequence length
NEW_CONFIG.update({"qkv_bias": True})  # Enable bias in query/key/value projections

# Initialize GPT model with the new configuration and set to evaluation mode
gpt = GPTModel(NEW_CONFIG)
gpt.eval()


# Helper function to assign pre-trained weights to model parameters
def assign(left, right):
    # Ensure the shapes match between existing parameter and new weight
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape} Right: {right.shape}")
    # Convert numpy array to PyTorch parameter tensor
    return torch.nn.Parameter(torch.tensor(right))


import numpy as np


def load_weights_into_gpt(gpt, params):
    """
    Load pre-trained GPT-2 weights into our custom GPT model.

    This function maps the OpenAI GPT-2 parameter structure to our model's architecture.
    The OpenAI model uses different naming conventions and weight organizations that
    need to be translated to match our implementation.

    Args:
        gpt: Our GPTModel instance to load weights into
        params: Dictionary of pre-trained parameters from OpenAI GPT-2
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained weights into our model and move to device
load_weights_into_gpt(gpt, params)
gpt.to(device)

# Example usage: Generate text using the loaded model
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
