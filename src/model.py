import torch
import torch.nn as nn

class TextGenerator(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)
    
