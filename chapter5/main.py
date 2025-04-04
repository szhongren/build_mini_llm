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

"""
look at how to assess text quality generated by a generative model by calculating a text generation loss

steps:
1. use vocabulary to map text to token IDs
2. get high dimensional probablility row vector for each token with softmax
3. locate the index position of the token ID in the row vector with the highest probability using argmax
4. get all predicted token IDs as the indices of the highest probability token IDs
5. map the predicted token IDs back to text

"""
# "Every effort moves"
# "I really like"
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])

# targets are one token later than the inputs, this is so we can teach the model to predict the next token
# "effort moves you"
# "really like chocolate"
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)
# returns (2, 3, 50257), which means 2 batches, 3 tokens, and 50257 vocab size

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(f"Token IDs: {token_ids}")

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# targets does not match the output because it's not been trained yet
# we can now evaluate the performance of the model's generated text with a loss function, so we can measure the quality of the generated text and also use it to implement the training function

text_idx = 0
# select the first text, then all 3 tokens, and then the target token IDs
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print(f"Text 1: {target_probas_1}")

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print(f"Text 2: {target_probas_2}")

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(f"Log probability: {log_probas}")

avg_log_probas = torch.mean(log_probas)
print(f"Avg log probability: {avg_log_probas}")

neg_avg_log_probas = avg_log_probas * -1
print(f"Negative avg log probability: {neg_avg_log_probas}")

# calculating loss involves following steps
# 1. get logits
# 2. get the probabilities with softmax
# 3. get the target probabilities
# 4. get the log probabilities
# 5. get the mean of the log probabilities
# 6. get the negative of the mean log probabilities

# these steps are already combined in the cross_entropy loss function, so we can use that

print(f"Logits shape: {logits.shape}")
print(f"Targets shape: {targets.shape}")
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print(f"Flattened logits shape: {logits_flat.shape}")
print(f"Flattened targets shape: {targets_flat.shape}")

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(f"Cross entropy loss: {loss}")

# perplexity is a measure of how well a probability distribution predicts a sample
# often used to evaluate language models as well
# it is the exponentiation of the cross entropy loss

perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity}")

# 5.1.3 calculating training and validation set losses
# here we will divide the dataset into training and validation sets, and calculate the training and validation losses
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(f"Total characters: {total_characters}")
print(f"Total tokens: {total_tokens}")

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from chapter2.gpt_dataset import create_dataloader_v1

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

print("Train loader")
for x, y in train_loader:
    print(x.shape, y.shape)

print("Validation loader")
for x, y in val_loader:
    print(x.shape, y.shape)

from .util import calc_loss_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print(f"Train loss: {train_loss}")
print(f"Validation loss: {val_loss}")
