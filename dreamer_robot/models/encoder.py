import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, image_size=64, embed_dim=128, depth=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,        depth,   4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(depth,  2*depth,   4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2*depth, 4*depth,  4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(4*depth, 8*depth,  4, stride=2, padding=1), nn.SiLU(),
        )
        s = image_size // 16          # 64 → 32 → 16 → 8 → 4
        self.proj = nn.Linear(8 * depth * s * s, embed_dim)

    def forward(self, x):
        # x: (B, T, H, W, 3) uint8
        B, T, H, W, C = x.shape
        x = x.float() / 255.0 - 0.5
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)   # (B*T, 3, H, W)
        x = self.conv(x)
        x = self.proj(x.flatten(1))
        return x.reshape(B, T, -1)                            # (B, T, embed_dim)
