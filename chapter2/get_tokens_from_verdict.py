import re
import os


def get_tokens_from_verdict():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(current_dir, "..", "the-verdict.txt"), "r", encoding="utf-8"
    ) as f:
        raw_text = f.read()

    print(f"Total number of characters: {len(raw_text)}")
    print(raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed
