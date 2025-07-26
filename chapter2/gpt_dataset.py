import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        """
        Creates a dataset for training GPT models by breaking text into input-target pairs.
        
        Think of this like creating flashcards where:
        - The front of each card shows some words
        - The back shows what word should come next
        
        Args:
            text: The raw text we want to train on (like a book or article)
            tokenizer: Tool that converts words into numbers the computer can understand
            max_length: How many words (tokens) to show the model at once
            stride: How many positions to skip when creating the next training example
        """
        # Create empty lists to store our training examples
        # Think of these as two stacks of flashcards
        self.input_ids = []    # Stack 1: "What you see" (the question)
        self.target_ids = []   # Stack 2: "What should come next" (the answer)

        # Convert the entire text into a list of numbers (tokens)
        # Each word/punctuation becomes a unique number the computer can work with
        # Example: "Hello world!" might become [15496, 995, 0]
        token_ids = tokenizer.encode(text)
        
        # Create training examples by sliding a window across the text
        # We stop before the end because we need space for the target
        for i in range(0, len(token_ids) - max_length, stride):
            
            # Get a chunk of tokens for input (what the model sees)
            # This is like showing the model: "The cat sat on the"
            # Starting at position i, take max_length tokens
            input_chunk = token_ids[i : i + max_length]
            
            # Get the target chunk (what the model should predict)
            # This is shifted by 1 position: "cat sat on the mat"
            # The model learns: given "The cat sat on the", predict "cat sat on the mat"
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            
            # Convert the number lists into PyTorch tensors (special arrays for AI)
            # Add this training pair to our collection
            self.input_ids.append(torch.tensor(input_chunk))    # The question
            self.target_ids.append(torch.tensor(target_chunk))  # The answer

    def __len__(self):
        """
        Returns how many training examples we have.
        Like counting how many flashcards are in our stack.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Gets one training example (flashcard) from our collection.
        
        When someone asks for flashcard number 5, we return:
        - The question (input): what the model should read
        - The answer (target): what the model should predict
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text,
    batch_size=4,        # How many examples to process at once (like studying 4 flashcards together)
    max_length=256,      # Maximum words per training example
    stride=128,          # How much to shift when creating the next example
    shuffle=True,        # Mix up the order of examples (like shuffling flashcards)
    drop_last=True,      # If the last batch is incomplete, skip it
    num_workers=0,       # How many helper processes to use (0 = main process only)
):
    """
    Creates a DataLoader that efficiently feeds training examples to our model.
    
    Think of this as an organized way to:
    1. Take our big text
    2. Break it into training examples (flashcards)
    3. Group examples into batches (study groups)
    4. Present them to the model in an organized way
    """
    # Get the GPT-2 tokenizer (the tool that converts words to numbers)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create our dataset (collection of all flashcards)
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    
    # Create a DataLoader (organized way to present flashcards to the model)
    # This handles batching, shuffling, and parallel processing
    dataloader = DataLoader(
        dataset,                    # Our collection of flashcards
        batch_size=batch_size,      # How many flashcards to show at once
        shuffle=shuffle,            # Whether to mix up the order
        drop_last=drop_last,        # Whether to skip incomplete final batch
        num_workers=num_workers,    # How many helper processes to use
    )
    return dataloader
