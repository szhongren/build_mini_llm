import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # crop current context to the context size
        with torch.no_grad():
            logits = model(idx_cond)  # forward pass to get logits

        logits = logits[:, -1, :]  # get the last time step
        probas = torch.softmax(logits, dim=-1)  # apply softmax to get probabilities
        idx_next = torch.argmax(
            probas, dim=-1, keepdim=True
        )  # get next token, the index with the highest probability
        idx = torch.cat((idx, idx_next), dim=1)  # append to the sequence

    return idx  # return input + generated tokens
