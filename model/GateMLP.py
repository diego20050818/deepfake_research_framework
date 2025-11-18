# -----------------------------
# Gated MLP (repr)
# -----------------------------
import torch
import torch.nn as nn

class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D]
        h = self.act(self.fc1(x))
        g = torch.sigmoid(self.gate(x))
        h = h * g
        h = self.dropout(h)
        h = self.fc2(h)
        return h