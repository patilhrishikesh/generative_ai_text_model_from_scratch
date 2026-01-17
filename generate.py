import torch

from src.preprocess import load_and_clean_text
from src.tokenizer import create_vocab
from src.model import TextGenerator


def generate_text(
    model,
    start_text,
    char_to_idx,
    idx_to_char,
    length=200,
    sequence_length=10
):
    model.eval()
    generated = start_text

    for _ in range(length):
        # Take last sequence_length characters
        input_seq = generated[-sequence_length:]

        # Convert characters to indices
        input_idx = [char_to_idx[ch] for ch in input_seq]
        input_tensor = torch.tensor([input_idx], dtype=torch.long)

        # Predict next character
        with torch.no_grad():
            output = model(input_tensor)

        next_idx = torch.argmax(output, dim=1).item()
        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated

if __name__ == "__main__":
    text = load_and_clean_text("data/text.txt")
    char_to_idx, idx_to_char = create_vocab(text)

    vocab_size = len(char_to_idx)
    model = TextGenerator(vocab_size, embed_dim=16, hidden_dim=64)

    model.load_state_dict(torch.load("model.pth"))

    output = generate_text(
        model,
        start_text="machine ",
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char
    )

    print(output)
