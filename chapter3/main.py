# 3.1 the problem with modeling long sequences

"""
we can't simply translate word by word because some words/phrases are dependent on words that come before or after.
This poses challenges in translation, especially with idiomatic expressions and context-sensitive meanings.

to address this problem, we commonly use deep neural networks with encoder and decoder submodules.

encoder: processes the input sequence and generates a context vector that captures the meaning of the entire sequence.
decoder: takes the context vector and generates the output sequence.

before transformers, we used RNNs and LSTMs

RNNs are a type of neural network where the output from the previous step is fed as input to the current step.
LSTMs are a type of RNN that can learn long-term dependencies. They are capable of learning order dependence in sequence prediction problems.

A big problem with RNNs and LSTMs is that they process the input sequence one step at a time, which makes them slow and difficult to parallelize.
Additionally, for RNNs, the RNN cannot directly access earlier hidden states, which makes it difficult to learn long-range dependencies as it has to rely solely on the current hidden state, which can lead to a loss of context.
"""

# 3.2 capturing data dependencies with attention mechanisms

"""
Researchers developed the Bahdanau attention mechanism to address the limitations of RNNs and LSTMs.

This mechanism allows the model to focus on relevant parts of the input sequence, improving the translation quality.
Additionally, it facilitates parallelization, making training faster compared to RNNs and LSTMs.

The attention mechanism computes a set of attention scores based on the current decoder state and the encoder outputs,
allowing for dynamic re-weighting of the encoder outputs based on their relevance to the current decoding step.

The self-attention mechanism is a key part of the architecture, enabling the model to learn relationships between all words in the input sequence simultaneously.
This transformation has significantly enhanced the performance of sequence-to-sequence tasks, such as machine translation.

"""

# 3.3 Attending to different parts of the input with self-attention

"""
The goal of self-attention is to compute a set of context vectors that capture the relationships between each word in the input sequence, enabling the model to weigh their importance when producing the output sequence.

Each context vector is with respect to a specific word in the input sequence. Their purpose is to create enriched representations of the input words by considering their relationships with other words in the sequence.
"""
import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)


def calculate_attention_scores_2(inputs):
    # calculate attention scores by taking the dot product of each input with a query vector
    query = inputs[1]
    attention_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attention_scores_2[i] = torch.dot(x_i, query)
    print(f"\n=== Attention Scores Calculation ===")
    print(f"Attention scores: {attention_scores_2}")
    return attention_scores_2


attention_scores_2 = calculate_attention_scores_2(inputs)

# normalize, so we get attention weights that sum to 1
# this is a useful convention for interpretation and maintaining numerical stability
attention_weights_2_tmp = attention_scores_2 / attention_scores_2.sum()
print(f"\nNormalized attention weights: {attention_weights_2_tmp}")
print(f"Sum: {attention_weights_2_tmp.sum():.4f}")


# in practice, we use the softmax function to normalize the attention scores because it handles extreme values better
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attention_weights_2_naive = softmax_naive(attention_scores_2)
print(f"\nNaive softmax attention weights: {attention_weights_2_naive}")
print(f"Sum: {attention_weights_2_naive.sum():.4f}")


def calculate_attention_weights_2(attention_scores_2):
    # but really, don't roll your own softmax
    attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
    print(f"\nPyTorch softmax attention weights: {attention_weights_2}")
    print(f"Sum: {attention_weights_2.sum():.4f}")
    return attention_weights_2


def calculate_context_vector_2(inputs, attention_weights_2):
    # final step, calculate context vector by multiplying the attention weights with the inputs, then summing
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attention_weights_2[i] * x_i
    print(f"\nComputed context vector: {context_vec_2}")
    print("-" * 50)


def calculate_attention_scores_loop(inputs):
    attention_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attention_scores[i][j] = torch.dot(x_i, x_j)
    print(f"\n=== Attention Scores (Loop) ===")
    print(f"Attention scores:\n{attention_scores}")
    return attention_scores


def calculate_attention_scores_mat_mul(inputs):
    attention_scores = inputs @ inputs.T
    print(f"\n=== Attention Scores (Matrix) ===")
    print(f"Attention scores:\n{attention_scores}")
    return attention_scores


attention_weights_2 = calculate_attention_weights_2(attention_scores_2)

calculate_context_vector_2(inputs, attention_weights_2)

attention_scores = calculate_attention_scores_mat_mul(inputs)

# normalize over last dimension, which in this case means it will softmax across each column so that the values in each row sum to 1
attention_weights = torch.softmax(attention_scores, dim=-1)
print(f"\nAttention weights:")
print(f"{attention_weights}")
print(f"Row sums: {attention_weights.sum(dim=-1)}")

context_vectors = attention_weights @ inputs
print(f"\nContext vectors:")
print(f"{context_vectors}")
print("-" * 50)

# 3.4 Implementing self-attention with trainable weights

"""
we are using scaled dot-product attention

the only difference from above is that we introduce weight matrices that are updated during model training. These are crucial so that the model can learn to produce good context vectors.

there are 3 weight matrices: W_q, W_k, and W_v
W_q: query weight matrix
W_k: key weight matrix
W_v: value weight matrix
"""

x_2 = inputs[1]
d_in = inputs.shape[1]  # input embedding size
d_out = 2  # output embedding size

# init weight matrices, use requires_grad=False to avoid updating them during training
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(f"Query: {query_2}")

# still need to get key and value for all inputs to calculate context vector for 2
keys = inputs @ W_key
values = inputs @ W_value

print(f"Keys shape: {keys.shape}")
print(f"Values shape: {values.shape}")

# then, calculate attention scores, this is different because we projected the input into a lower dimensional space before we calculated the attention scores
keys_2 = keys[1]
attention_score_22 = query_2.dot(keys_2)
print(f"Attention score_22: {attention_score_22}")

attention_scores_2 = query_2 @ keys.T
print(f"Attention scores_2: {attention_scores_2}")

# now, calculate the attention weights, by taking square root and then using softmax
# this improves training performance by avoiding small gradients
d_k = keys.shape[-1]
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)
print(f"Attention weights_2: {attention_weights_2}")

# finally, calculate the context vector
context_vector_2 = attention_weights_2 @ values
print(f"Context vector_2: {context_vector_2}")

# query, key, value are borrowed from information retrieval and databases, corresponding to the search term, the document, and the information in the document, respectively.

from .self_attention_v1 import SelfAttention_v1

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(f"Self-attention v1 output: {sa_v1(inputs)}")

from .self_attention_v2 import SelfAttention_v2

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(f"Self-attention v2 output: {sa_v2(inputs)}")

# print(sa_v1.W_query)
# print(sa_v2.W_query.weight.T)
# print(torch.nn.Parameter(sa_v2.W_query.weight.T))
# print(sa_v1.W_key)
# print(sa_v2.W_key.weight.T)
# print(torch.nn.Parameter(sa_v2.W_key.weight.T))
# print(sa_v1.W_value)
# print(sa_v2.W_value.weight.T)
# print(torch.nn.Parameter(sa_v2.W_value.weight.T))

# copy weights from v2 to v1, the values are different because we initialized v1 with random values, but importantly, the output is the same
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

print(f"Self-attention v1 output: {sa_v1(inputs)}")

# 3.5 hiding future words with causal attention

"""
We want to predict the next word without knowing the next word, so we need to hide future words. To do this, we use causal attention or masked attention.

In causal attention, the attention weights are masked so that the model cannot attend to future words.
"""

# calculate attention weights without masking
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attention_weights)

# create a simple mask to hide future words
context_length = attention_scores.shape[0]
# tril creates a lower triangular matrix with ones on the diagonal and below
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# apply the mask to the attention weights
masked_simple = attention_weights * mask_simple
print(masked_simple)

# renormalize the masked attention weights
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

"""
A more effecient way to get the masked weights is to mask the attention scores with negative infinity before applying softmax
"""

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# and apply softmax
attention_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attention_weights)

"""
dropout is a technique where ramdomly selected hidden layer units are ignored during training that helps prevent overfitting by ensuring that a model does not become overly reliant on any specific set of hidden layer units

only used during training and disabled afterwards
"""
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

"""
to compensate for half the values dropping out, the rest of the values are scaled up by 2 to keep the expected output the same
"""

torch.manual_seed(123)
print(dropout(attention_weights))

""""
simple compact causal attention
"""

# simmulate bath inputs
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

torch.Size([2, 6, 3])

from .causal_attention import CausalAttention

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(f"Context vectors shape: {context_vecs.shape}")

# 3.6 extending single-head attention to multi-head attention

"""
multi head attention refers to dividing the attention mechanism into multiple heads, each operating independently

we will stack multiple causal attention modules, each with its own set of weight matrices, as a first pass

then, we will implement the same multi head attention module in a more complicated but more efficient way

2 single head modules, 2 sets of each weight matrix, resulting in 2 attention weight matrices and 2 context vectors

finally, we combine the 2 context vectors into a single context vector
"""

from .multi_head_attention import MultiHeadAttentionWrapper

torch.manual_seed(123)
context_lenghth = batch.shape[1]  # this is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_lenghth, 0.0, 2)

context_vecs = mha(batch)

print(context_vecs)
print(f"context vecs shape: {context_vecs.shape}")


def example():
    a = torch.tensor(
        [
            [
                [
                    [0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340],
                ],
                [
                    [0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786],
                ],
            ]
        ]
    )
    # now we perform a batched matrix multiplication between the tensor and a view of the tensor where we transpose the last two dimensions
    b = a @ a.transpose(2, 3)
    print(f"a: {a}")
    print(f"a.shape: {a.shape}")
    print(f"a.transpose(2, 3): {a.transpose(2, 3)}")
    print(f"a.transpose(2, 3).shape: {a.transpose(2, 3).shape}")
    print(f"b: {b}")
    print(f"b.shape: {b.shape}")

    first_head = a[0, 0, :, :]
    first_result = first_head @ first_head.T
    print(f"first_head: {first_head}")
    print(f"first_result: {first_result}")
    second_head = a[0, 1, :, :]
    second_result = second_head @ second_head.T
    print(f"second_head: {second_head}")
    print(f"second_result: {second_result}")


example()

from .multi_head_attention import MultiHeadAttention

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(f"Context vectors: {context_vecs}")
print(f"Context vectors shape: {context_vecs.shape}")
