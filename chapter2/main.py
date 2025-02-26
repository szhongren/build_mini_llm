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

# 2.5 bype pair encoding
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

# 2.6 data sampling with a sliding windowh
from gpt_dataset import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"X: {x}")
print(f"y:      {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(context, "----->", desired)
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader2 = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter2 = iter(dataloader2)
first_batch2 = next(data_iter2)
print(first_batch2)
