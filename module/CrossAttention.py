# -----------------------------
# CrossAttentionCombination (simplified as in analysis)
# -----------------------------
import torch
import torch.nn as nn

class CrossAttentionCombination(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super().__init__()
        self.projA = nn.Linear(dim_a, out_dim)
        self.projB = nn.Linear(dim_b, out_dim)
        self.projOut = nn.Linear(out_dim*2, out_dim)

    def forward(self, x1, x2):
        """
        x1: [B, dim_a]
        x2: [B, dim_b]
        """

        a = self.projA(x1)  # [B, out]
        b = self.projB(x2)  # [B, out]
        a_s = torch.softmax(a, dim=-1)
        b_s = torch.softmax(b, dim=-1)
        
        # cross weights
        part1 = a_s * b  # [B, out]
        part2 = b_s * a  # [B, out]
        out = torch.cat([part1, part2], dim=-1)
        return self.projOut(out)  # [B, out]