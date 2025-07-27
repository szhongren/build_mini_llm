# 6.1 Different categories of fine-tuning

"""
2 most common ways to fine-tune language models:
* instruction fine-tuning
* classification fine-tuning

classification is generally more specialized, and instruction is more general
instruction is usually better for models that need to handle a variety of tasks based on complex user instructions
classification is ideal for projects that require precise categorization of data
"""

# 6.2 Preparing the dataset
import urllib.request
import zipfile
import os
from pathlib import Path

from .dataset import SpamDataset

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


"""
get and unzip the data
"""


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download " "and extraction.")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

import pandas as pd

"""
load into dataframe
"""
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df)
print(df["Label"].value_counts())


"""
balance the dataset so we get equal numbers of ham and spam
"""


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})


"""
split randomly into 70% training, 10% validation, 20% testing
"""


def random_split(df, train_frac, validation_frac):

    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

# 6.3 Creating data loaders

"""
we will develop pytorch data loaders that are similar to those that we created when working with text data

we used a sliding window technique to create uniformly sized text chunks, then batched for more efficient model training. Each chunk is a single training instance, but now we have different text messages of different lengths, so we can't use the sliding window technique. Here, we can either pad or truncate, padding is better because we avoid losing information

we can use <|endoftext|> as a padding token
"""

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)

print(train_dataset.max_length)
print(train_dataset.encoded_texts[:2])

val_dataset = SpamDataset(
    csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)

"""
the following code creates the training, validation, and test set data loaders that load the messages and labels in batches of size 8
"""

import torch
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
"""
check the batch size and the number of batches in each dataset
"""
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# 6.4 Initializing a model with pretrained weights

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
