# 2.1 GET TOKENS - Breaking text into individual words/pieces
# Think of this like cutting a sentence into individual word cards
from .get_tokens_from_verdict import get_tokens_from_verdict

print("=== Tokenization Demo ===")
# Get all the word pieces from our text file
# This breaks "Hello world!" into ["Hello", "world", "!"]
preprocessed = get_tokens_from_verdict()
print(f"Total tokens: {len(preprocessed)}")  # How many word pieces we found
print(f"First 30 tokens: {preprocessed[:30]}")  # Show first 30 pieces

# 2.2 CREATE VOCABULARY - Making a dictionary of all unique words
# Think of this like creating an index at the back of a book
# Every unique word gets assigned a number (like page numbers)
all_words = sorted(set(preprocessed))  # Get unique words and sort them alphabetically
vocab_size = len(all_words)  # Count how many different words we have
print(f"\n=== Vocabulary Creation ===")
print(f"Vocabulary size: {vocab_size}")

# Create a mapping: word -> number
# Example: {"hello": 0, "world": 1, "the": 2, ...}
# This lets us convert words to numbers the computer can work with
vocab = {token: integer for integer, token in enumerate(all_words)}

print("\nFirst 50 vocabulary items:")
# Show the first 50 word-to-number mappings
for i, item in enumerate(vocab.items()):
    print(f"  {item}")  # Print each (word, number) pair
    if i >= 49:  # Stop after showing 50 items
        break

# 2.3 CREATE SIMPLE TEXT TOKENIZER - A tool to convert between words and numbers
# Think of this like a translator that speaks both "human language" and "computer language"
from .simple_tokenizer_v1 import SimpleTokenizerV1

# Create our translator using the vocabulary we just made
tokenizer = SimpleTokenizerV1(vocab)
text = (
    """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
)
print("\n=== Simple Tokenizer Test ===")
print(f"Text: {text}")

# Convert text to numbers (encode) - like translating English to computer language
ids = tokenizer.encode(text)
print(f"Encoded IDs: {ids}")  # Show the numbers

# Convert numbers back to text (decode) - translate back to English
print(f"Decoded text: {tokenizer.decode(ids)}")
print("-" * 50)

# 2.4 ADDING SPECIAL CONTEXT TOKENS - Adding special "control" words
# Think of these like punctuation marks that mean something special
# <|endoftext|> = "This document is finished"
# <|unk|> = "Unknown word" (for words not in our vocabulary)
from .simple_tokenizer_v2 import SimpleTokenizerV2

# Start with our original vocabulary
all_tokens = sorted(list(set(preprocessed)))
# Add special tokens - like adding new punctuation marks to our language
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# Recreate the vocabulary with the new special tokens included
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))  # Show total vocabulary size

# Show the last 5 items (which include our special tokens)
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# Test the special tokens by joining two sentences
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
# Put the special "end of text" marker between them
text = " <|endoftext|> ".join((text1, text2))
print(text)

# Test our improved tokenizer that understands special tokens
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))  # Convert to numbers
print(tokenizer.decode(tokenizer.encode(text)))  # Convert back to text

# 2.5 BYTE PAIR ENCODING - Using GPT-2's professional tokenizer
# Think of this like upgrading from a basic dictionary to a professional one
# GPT-2's tokenizer is much smarter and handles unknown words better
import tiktoken

# Get the same tokenizer that GPT-2 uses (much more sophisticated)
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)

# Convert text to numbers using the professional tokenizer
# allowed_special tells it that <|endoftext|> is okay to use
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)  # Show the numbers
strings = tokenizer.decode(integers)  # Convert back to text
print(strings)

# 2.6 DATA SAMPLING WITH A SLIDING WINDOW - Creating training examples
# Think of this like making flashcards from a book by sliding a window across pages
from .gpt_dataset import create_dataloader_v1
import torch
import os

# Find and read our text file (the-verdict.txt)
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(
    os.path.join(current_dir, "..", "the-verdict.txt"), "r", encoding="utf-8"
) as f:
    raw_text = f.read()  # Read the entire book into memory

# Convert the entire book to numbers
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))  # Show how many tokens (number-words) we have

# Take a sample starting from position 50 (skip the first 50 tokens)
enc_sample = enc_text[50:]

# Create a simple example with context_size = 4
# This means "show 4 words, predict the next 4"
context_size = 4
x = enc_sample[:context_size]  # Input: first 4 tokens
y = enc_sample[1 : context_size + 1]  # Target: tokens 2-5 (shifted by 1)
print(f"X: {x}")  # What the model sees
print(f"y:      {y}")  # What the model should predict

# Show how the model learns step by step
# Given 1 word, predict the next; given 2 words, predict the next, etc.
for i in range(1, context_size + 1):
    context = enc_sample[:i]  # Take i words as context
    desired = enc_sample[i]   # The next word is what we want to predict
    # Convert numbers back to words to see what's happening
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

# Create a dataloader to organize our training examples
# Think of this as an assembly line that packages flashcards into study groups
dataloader = create_dataloader_v1(
    raw_text,        # The book we're learning from
    batch_size=1,    # Study 1 flashcard at a time
    max_length=4,    # Each flashcard shows 4 words
    stride=1,        # Move 1 word over for the next flashcard (maximum overlap)
    shuffle=False    # Keep them in order for now
)

# Get an iterator (like a conveyor belt) to get batches one at a time
data_iter = iter(dataloader)
first_batch = next(data_iter)  # Get the first batch (1 training example)
print(first_batch)

second_batch = next(data_iter)  # Get the second batch
print(second_batch)

# Create another dataloader with different settings
dataloader2 = create_dataloader_v1(
    # batch_size -> number of sequences (flashcards) to study together
    # max_length -> number of words on each flashcard
    # stride -> how many words to skip when making the next flashcard
    raw_text,
    batch_size=8,    # Study 8 flashcards together (more efficient)
    max_length=4,    # Still 4 words per flashcard
    stride=4,        # Skip 4 words between flashcards (no overlap)
    shuffle=False,   # Keep in order
)
data_iter2 = iter(dataloader2)
first_batch2 = next(data_iter2)  # Get 8 training examples at once
print(first_batch2)

# 2.7 SIMPLE EMBEDDING EXAMPLE - Converting numbers to vectors
# Think of this like giving each word a unique "fingerprint" of numbers
# Instead of just "word = 5", we give it a pattern like [0.2, -0.1, 0.8]

test_input_ids = torch.tensor([2, 3, 5, 1])  # Some example word numbers
vocab_size = 6      # We have 6 different words total
output_dim = 3      # Each word gets a 3-number fingerprint
torch.manual_seed(123)  # Make results repeatable

# Create an embedding layer - like a lookup table of fingerprints
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)  # Show all the fingerprints

# Look up the fingerprint for word number 3
print(embedding_layer(torch.tensor([3])))

# Look up fingerprints for multiple words at once
print(embedding_layer(test_input_ids))

# 2.8 ENCODING WORD POSITIONS - Teaching the model about word order
# Think of this like adding timestamps to show "which word came first"
# The model needs to know that "dog bites man" is different from "man bites dog"

vocab_size = 50257  # GPT-2's full vocabulary size
output_dim = 256    # Each word gets a 256-number fingerprint (much richer!)
# Create embedding layer for word meanings
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4  # Work with 4-word sequences
# Get some real training data
data_loader = create_dataloader_v1(
    raw_text, 
    batch_size=8,           # Process 8 examples together
    max_length=max_length,  # 4 words per example
    stride=max_length,      # No overlap between examples
    shuffle=False
)
data_iter = iter(data_loader)
inputs, targets = next(data_iter)  # Get one batch
print(f"Token IDS:\n{inputs}")      # Show the word numbers
print(f"Inputs shape:\n{inputs.shape}")  # Shape: [8 examples, 4 words each]

# Convert word numbers to their meaning fingerprints
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)  # Shape: [8, 4, 256] - 8 examples, 4 words, 256-dim fingerprints

# Create position embeddings - fingerprints for "1st word", "2nd word", etc.
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# Create position fingerprints for positions 0, 1, 2, 3
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)  # Shape: [4, 256] - 4 positions, 256-dim fingerprints

# Combine word meaning + position information
# This is like saying "the word 'dog' in position 1" vs "the word 'dog' in position 3"
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)  # Shape: [8, 4, 256] - complete word+position fingerprints
