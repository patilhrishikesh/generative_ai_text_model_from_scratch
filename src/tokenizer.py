def create_vocab(text):
     # creates character level vocabulary mappings.
    chars = sorted(list(text))
     
    char_to_idx = {}
    idx_to_char = {}
     
    for idx, ch in enumerate(chars):
         char_to_idx[ch] = idx
         idx_to_char[idx] = ch
         
    return char_to_idx, idx_to_char