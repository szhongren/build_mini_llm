# 2.1 get tokens
from get_tokens_from_verdict import get_tokens_from_verdict

preprocessed = get_tokens_from_verdict()
print(len(preprocessed))
print(preprocessed[:30])

# 2.2 create vocabulary
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
vocab = {token: integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)

    if i >= 50:
        break

# 2.3 create simple text tokenizer
from simple_tokenizer_v1 import SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)
text = (
    """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# 2.4 adding special context tokens
from simple_tokenizer_v2 import SimpleTokenizerV2

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
