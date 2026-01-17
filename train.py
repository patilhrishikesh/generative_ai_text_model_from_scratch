import torch
import torch.nn as nn
import torch.optim as optim

from src.preprocess import load_and_clean_text
from src.tokenizer import create_vocab
from src.dataset import create_sequences
from src.model import TextGenerator

# Load data
text = load_and_clean_text("data/text.txt")
char_to_idx, idx_to_char = create_vocab(text)

# create training data
X, y = create_sequences(text, char_to_idx, sequence_length=10)

# convert to tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# model initialization
vocab_size = len(char_to_idx)
model = TextGenerator(vocab_size, embed_dim = 16, hidden_dim=64)

# loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("Vocabulary size:", vocab_size)
print("Training samples:", X.shape[0])
