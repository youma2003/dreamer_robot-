import torch
import torch.nn as nn


class ImageDecoder(nn.Module):
    """Symmetric mirror of CNNEncoder."""

    def __init__(self, latent_dim=256, hidden_dim=128, image_size=64, depth=24):
        super().__init__()
        self.s     = image_size // 16   # 4 for image_size=64
        self.depth = depth
        feat_dim   = hidden_dim + latent_dim

        self.proj = nn.Linear(feat_dim, 8 * depth * self.s * self.s)

        # 4 → 8 → 16 → 32 → 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8*depth, 4*depth, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(4*depth, 2*depth, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(2*depth,   depth, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(  depth,       3, 4, stride=2, padding=1),
        )

    def forward(self, h, z):
        # h: (B, T, hidden_dim),  z: (B, T, latent_dim)
        B, T, _ = h.shape
        feat = torch.cat([h, z], dim=-1)            # (B, T, feat_dim)
        x    = self.proj(feat)                       # (B, T, 8*d*s*s)
        x    = x.reshape(B * T, 8 * self.depth, self.s, self.s)
        x    = self.deconv(x)                        # (B*T, 3, H, W)
        C, H, W = x.shape[1:]
        return x.reshape(B, T, C, H, W)             # (B, T, 3, H, W)


class MLPHead(nn.Module):
    """2-layer MLP used for reward and continue predictions."""

    def __init__(self, feat_dim=384, out_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h, z):
        feat = torch.cat([h, z], dim=-1)
        return self.net(feat)                        # (B, T, out_dim)
