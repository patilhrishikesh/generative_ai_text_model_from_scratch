def create_sequences(text, char_to_idx, sequence_length = 10):
    # converts text into input-output pairs for training
    
    X = []
    y = []
    
    for i in range(len(text) - sequence_length):
        input_seq = text[i : i + sequence_length]
        target_char = text[i + sequence_length]
        
         # SAFETY CHECK
        if any(ch not in char_to_idx for ch in input_seq + target_char):
            continue
        
        X.append([char_to_idx[ch] for ch in input_seq])
        y.append(char_to_idx[target_char])
        
    return X, y