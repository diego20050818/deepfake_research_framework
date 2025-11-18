# -----------------------------
# Full Model
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

# import os
# import sys
# sys.path.append(os.getcwd())
from model.CrossAttention import CrossAttentionCombination
from model.DCTConvBlock import DCTConvBlock
from model.GateMLP import GatedMLP
from model.SSCA import SSCA
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://mirrors.aliyun.com/hugging-face-models'

class RINEPlusSSCA(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-large-patch14', proj_dim=1024, repr_dim=512, use_layers=None):
        """
        use_layers: list of indices of transformer blocks to use (1..n). If None uses all.
        """
        super().__init__()

        self.clip = CLIPVisionModel.from_pretrained(
            clip_model_name, 
            output_hidden_states=True,
            mirror="https://mirrors.aliyun.com/hugging-face-models"
        )

        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        # get n blocks
        self.num_blocks = len(self.clip.vision_model.encoder.layers)  # typically 24 for L/14
        self.hidden_dim = self.clip.config.hidden_size  # d
        if use_layers is None:
            self.use_layers = list(range(self.num_blocks))  # 0..n-1
        else:
            self.use_layers = use_layers
        self.n_use = len(self.use_layers)

        # Projection network Q1 (applied to each CLS from intermediate layers)
        # We will map each CLS (d) -> d' (proj_dim) per layer, then compute weighted sum
        self.proj_per_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ) for _ in range(self.n_use)])

        # Trainable Importance Estimator (A): we'll parameterize as a learnable matrix n x proj_dim
        self.A = nn.Parameter(torch.randn(self.n_use, proj_dim) * 0.01)

        # Q2 projection network after weighted sum
        self.q2 = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU()
        )

        # SSCA branch: works on mid-level feature maps; we need a lightweight conv stem to match dims
        # We'll extract the CLIP patch embedding outputs (not trivial). For prototype, we create a small conv stem to accept input image.
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )
        self.dct_block = DCTConvBlock(64, hidden_ch=128)
        self.ssca = SSCA(128, reduction=16)
        # pool sscha output to vector
        self.ssca_pool = nn.AdaptiveAvgPool2d(1)

        # Cross-attention combination mapping
        self.cross_comb = CrossAttentionCombination(dim_a=proj_dim, dim_b=128, out_dim=repr_dim)

        # Representation head
        self.gated = GatedMLP(repr_dim, hidden_dim=repr_dim*2)
        self.class_head = nn.Linear(repr_dim, 1)  # logits
        # a representation output for contrastive loss
        self.repr_out = nn.Linear(repr_dim, repr_dim)

    def forward(self, images):
        """
        images: [B, 3, H, W] assumed 224x224
        returns: logits [B], repr for contrastive [B, repr_dim]
        """
        B = images.shape[0]
        # 1) get CLIP hidden states: set model in eval (frozen)
        # CLIP vision model expects pixel_values; use its feature extractor externally in dataloader; here we assume images are normalized as CLIP expects.
        clip_out = self.clip(pixel_values=images)  # returns last_hidden_state and hidden_states
        # hidden_states: tuple len n+1 (embedding outputs and each block output)
        hidden = clip_out.hidden_states  # list of tensors [B, p+1, d]
        # Extract CLS tokens (index 0) from selected layers
        cls_list = []
        for i, layer_idx in enumerate(self.use_layers):
            # in HuggingFace CLIP, hidden[layer_idx+1] correspond to output after that block (first is embeddings)
            h = hidden[layer_idx + 1][:, 0, :]  # CLS token (B, d)
            cls_list.append(h)
        # per-layer project and stack
        proj_feats = []
        for i, cls in enumerate(cls_list):
            proj_feats.append(self.proj_per_layer[i](cls))  # (B, proj_dim)
        proj_stack = torch.stack(proj_feats, dim=1)  # (B, n_use, proj_dim)
        # importance scores via softmax across layers
        alpha = torch.softmax(self.A, dim=0)  # (n_use, proj_dim)
        alpha = alpha.unsqueeze(0)  # (1, n_use, proj_dim)
        weighted = (proj_stack * alpha).sum(dim=1)  # (B, proj_dim)
        q2_out = self.q2(weighted)  # (B, proj_dim) -> this is z_clip style vector
        z_clip = q2_out

        # 2) SSCA branch: process raw images through stem & DCT-like block & SSCA
        x_stem = self.stem_conv(images)  # [B, 64, H/2, W/2]
        dct_feat = self.dct_block(x_stem)  # [B, 128, 1, 1]
        # optionally expand to spatial for SSCA: create a small feature map by tiling
        small_map = dct_feat.expand(-1, -1, 14, 14)  # [B, 128, 14, 14] (prototype)
        ssca_out = self.ssca(small_map)  # [B, 128, 14, 14]
        ssca_vec = self.ssca_pool(ssca_out).view(B, 128)  # [B, 128]
        z_dct = ssca_vec

        # 3) CrossAttentionCombination
        z_repr = self.cross_comb(z_clip, z_dct)  # [B, repr_dim]
        # 4) Gated MLP
        z_repr = self.gated(z_repr)  # [B, repr_dim]
        logits = self.class_head(z_repr).squeeze(1)  # [B]
        rep_for_contrast = self.repr_out(z_repr)  # [B, repr_dim]
        return logits, rep_for_contrast