"""
novel_model_alpha03.py

RA-TKD DeiT-style student architecture (alpha = 0.3 configuration) for
multi-label chest X-ray pathology classification.

Contains a convolutional stem, patch embedding, a region-aware token
selector initialised with a blend weight of alpha_init = 0.3 (a moderate
mix of the fixed anatomical prior and learned token relevance), eight
DeiT-style transformer encoder blocks, an attention-pooling module, and a
14-label classification head.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvStem(nn.Module):
    """Two-layer convolutional stem for initial feature extraction."""


    def __init__(self, in_ch=3, hidden_ch=32, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


    def forward(self, x):
        return self.net(x)



class PatchEmbed(nn.Module):
    """Converts convolutional features into patch-token embeddings."""


    def __init__(self, in_ch=64, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class RegionAwareTokenSelector(nn.Module):
    """Selects high-priority tokens and pools the remaining tokens into summaries."""


    def __init__(
        self,
        embed_dim=192,
        grid_size=14,
        keep_tokens=98,
        summary_tokens=14,
        alpha_init=0.3
    ):
        super().__init__()
        self.keep_tokens = keep_tokens
        self.summary_tokens = summary_tokens


        # Parameterise alpha in logit space so it remains between zero and one.
        alpha_init = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self._alpha_raw = nn.Parameter(
            torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
        )


        # Predict learned token relevance scores.
        self.score_head = nn.Linear(embed_dim, 1)


        # Create a fixed central spatial prior over the 14 × 14 patch grid.
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size),
            indexing="ij"
        )
        dist = torch.sqrt((xs * 0.9) ** 2 + (ys * 1.2) ** 2)
        prior = 1.0 - dist
        prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-8)
        self.register_buffer("prior_map", prior.flatten())


    @property
    def alpha(self):
        return torch.sigmoid(self._alpha_raw)


    def forward(self, tokens):
        B, N, C = tokens.shape


        # Blend learned relevance with the fixed spatial prior.
        learned_scores = torch.sigmoid(self.score_head(tokens).squeeze(-1))
        prior_scores = self.prior_map.unsqueeze(0).expand(B, -1)
        alpha = self.alpha
        scores = alpha * prior_scores + (1 - alpha) * learned_scores


        # Select the highest-scoring tokens and retain their original token order.
        _, topk_idx = torch.topk(
            scores,
            self.keep_tokens,
            dim=1,
            largest=True,
            sorted=False
        )
        topk_idx, _ = torch.sort(topk_idx, dim=1)


        kept = torch.gather(
            tokens,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )


        # Use blended scores to scale retained tokens and preserve an alpha gradient path.
        kept_gate = torch.gather(scores, 1, topk_idx).unsqueeze(-1)
        kept = kept * (1.0 + kept_gate)


        # Collect lower-priority tokens for contextual summary construction.
        mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, topk_idx, False)


        low_tokens = tokens[mask].view(B, N - self.keep_tokens, C)


        if low_tokens.shape[1] < self.summary_tokens:
            pad = self.summary_tokens - low_tokens.shape[1]
            low_tokens = F.pad(low_tokens, (0, 0, 0, pad))


        # Mean-pool lower-priority tokens into compact summary tokens.
        chunks = torch.chunk(low_tokens, self.summary_tokens, dim=1)
        summary = torch.stack([c.mean(dim=1) for c in chunks], dim=1)


        out = torch.cat([kept, summary], dim=1)
        return out, scores



class MLP(nn.Module):
    """Feed-forward layer used within each transformer block."""


    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))



class TransformerBlock(nn.Module):
    """Pre-normalisation transformer encoder block."""


    def __init__(self, dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)


    def forward(self, x):
        # Apply residual self-attention and feed-forward updates.
        y = self.norm1(x)
        y, attn = self.attn(y, y, y, need_weights=True)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, attn



class AttentionPooling(nn.Module):
    """Produces a learned weighted representation of the image tokens."""


    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)


    def forward(self, x):
        weights = torch.softmax(self.proj(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights



class RATKDDeiT(nn.Module):
    """RA-TKD student with region-aware token selection and DeiT-style encoding."""


    def __init__(
        self,
        num_classes=14,
        embed_dim=192,
        depth=8,
        num_heads=3,
        keep_tokens=98,
        summary_tokens=14,
        alpha_init=0.3
    ):
        super().__init__()
        self.stem = ConvStem()
        self.patch_embed = PatchEmbed(in_ch=64, embed_dim=embed_dim)
        self.selector = RegionAwareTokenSelector(
            embed_dim,
            14,
            keep_tokens,
            summary_tokens,
            alpha_init=alpha_init
        )


        # Add one class token to the selected and summary image tokens.
        total_tokens = keep_tokens + summary_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens + 1, embed_dim)
        )


        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])


        self.norm = nn.LayerNorm(embed_dim)
        self.pool = AttentionPooling(embed_dim)


        # Fuse class-token and pooled-token features before output prediction.
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )


        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x, return_features=False):
        # Extract patch tokens and reduce them using the region-aware selector.
        x = self.stem(x)
        x = self.patch_embed(x)
        x, token_scores = self.selector(x)


        # Add class token and positional embeddings before transformer encoding.
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]


        features = {}
        attn_maps = []


        # Store intermediate and final representations for analysis or distillation.
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            attn_maps.append(attn)


            if i == 3:
                features["mid"] = x


        x = self.norm(x)
        features["final"] = x


        # Combine class-token and attention-pooled token features for prediction.
        cls_feat = x[:, 0]
        token_feat = x[:, 1:]
        pooled, pool_weights = self.pool(token_feat)
        fused = torch.cat([cls_feat, pooled], dim=1)
        logits = self.head(fused)


        # Return internal representations when required for distillation or visualisation.
        if return_features:
            return {
                "logits": logits,
                "features": features,
                "token_scores": token_scores,
                "attn_maps": attn_maps,
                "pool_weights": pool_weights,
            }


        return logits