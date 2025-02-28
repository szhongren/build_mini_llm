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
    print(f"Attention scores: {attention_scores_2}")
    return attention_scores_2


attention_scores_2 = calculate_attention_scores_2(inputs)

# normalize, so we get attention weights that sum to 1
# this is a useful convention for interpretation and maintaining numerical stability
attention_weights_2_tmp = attention_scores_2 / attention_scores_2.sum()
print(
    f"Attention weights: {attention_weights_2_tmp} - Sum: {attention_weights_2_tmp.sum()}"
)


# in practice, we use the softmax function to normalize the attention scores because it handles extreme values better
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attention_weights_2_naive = softmax_naive(attention_scores_2)
print(
    f"Attention weights (naive): {attention_weights_2_naive} - Sum: {attention_weights_2_naive.sum()}"
)


def calculate_attention_weights_2(attention_scores_2):
    # but really, don't roll your own softmax
    attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
    print(
        f"Attention weights: {attention_weights_2} - Sum: {attention_weights_2.sum()}"
    )
    return attention_weights_2


def calculate_context_vector_2(inputs, attention_weights_2):
    # final step, calculate context vector by multiplying the attention weights with the inputs, then summing
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attention_weights_2[i] * x_i
    print(f"Context vector: {context_vec_2}")


def calculate_attention_scores_loop(inputs):
    attention_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attention_scores[i][j] = torch.dot(x_i, x_j)
    print(f"Attention scores: {attention_scores}")
    return attention_scores


def calculate_attention_scores_mat_mul(inputs):
    attention_scores = inputs @ inputs.T
    print(f"Attention scores: {attention_scores}")
    return attention_scores


attention_weights_2 = calculate_attention_weights_2(attention_scores_2)

calculate_context_vector_2(inputs, attention_weights_2)

attention_scores = calculate_attention_scores_mat_mul(inputs)

# normalize over last dimension, which in this case means it will softmax across each column so that the values in each row sum to 1
attention_weights = torch.softmax(attention_scores, dim=-1)
print(f"Attention weights: {attention_weights} - Sum: {attention_weights.sum(dim=-1)}")

context_vectors = attention_weights @ inputs
print(f"Context vectors: {context_vectors}")

# 3.4 Implementing self-attention with trainable weights
