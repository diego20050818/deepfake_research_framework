import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self,d_model=768,bottleneck=64,dropout=0.1) -> None:
        super().__init__()
        self.down_proj = nn.Linear(d_model,bottleneck)
        self.ReLU = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck,d_model)
        self.dropout = dropout

    def forward(self,x,add_residual=True):
        residual = x
        down = self.down_proj(x)
        down = self.ReLU(down)
        down = F.dropout(down,p=self.dropout,training=self.training)
        up = self.up_proj(down)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        return output
