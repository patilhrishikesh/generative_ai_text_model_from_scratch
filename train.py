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

print("char_to_idx:")
print(char_to_idx)

print("\nidx_to_char:")
print(idx_to_char)

# create training data
X, y = create_sequences(text, char_to_idx, sequence_length=10)


# convert to tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# model initialization
vocab_size = len(char_to_idx)
model = TextGenerator(vocab_size, embed_dim = 16, hidden_dim=64)
print("Max token index in X:", X.max().item())
print("Vocab size:", vocab_size)
# loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# print("Vocabulary size:", vocab_size)
# print("Training samples:", X.shape[0])

epochs = 80

for epoch in range(epochs):
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = loss_fn(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss:{loss.item():.4f}")
        
        

torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
