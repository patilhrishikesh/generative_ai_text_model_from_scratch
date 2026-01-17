def create_vocab(text):
    # Get unique characters
    chars = sorted(set(text))

    # Proper sequential indexing
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

    return char_to_idx, idx_to_char
