"""
Smoke test — runs on CPU with fake data, no dataset required.
Usage: python test_shapes.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import numpy as np

from dreamer_robot.models.encoder      import CNNEncoder
from dreamer_robot.models.rssm         import RSSM
from dreamer_robot.models.heads        import ImageDecoder, MLPHead
from dreamer_robot.models.actor_critic import Actor, Critic
from dreamer_robot.training.losses     import kl_loss, recon_loss, lambda_returns
from dreamer_robot.utils.replay_buffer import ReplayBuffer

# ── hyper-parameters (must match Colab notebook) ────────────────────────── #
B          = 2
T          = 8
HORIZON    = 10
image_size = 64
embed_dim  = 128
hidden_dim = 128
num_cats   = 16
cat_size   = 16
latent_dim = 256   # num_cats * cat_size
action_dim = 7
feat_dim   = 384   # hidden_dim + latent_dim
chunk_len  = 24
BATCH      = 4

# ── fake inputs ─────────────────────────────────────────────────────────── #
images  = torch.randint(0, 256, (B, T, image_size, image_size, 3), dtype=torch.uint8)
actions = torch.randn(B, T, action_dim)

# ── encoder ─────────────────────────────────────────────────────────────── #
encoder = CNNEncoder(image_size=image_size, embed_dim=embed_dim)
embeds  = encoder(images)
assert embeds.shape == (B, T, embed_dim), \
    f"encoder output: expected (2, 8, 128), got {embeds.shape}"
print(f"✅  encoder output    : {tuple(embeds.shape)}")

# ── rssm observe ────────────────────────────────────────────────────────── #
rssm = RSSM(embed_dim, hidden_dim, action_dim, num_cats, cat_size)
h, z, post_logits, prior_logits = rssm.observe(embeds, actions)

assert h.shape == (B, T, hidden_dim), \
    f"rssm h: expected (2, 8, 128), got {h.shape}"
print(f"✅  rssm h            : {tuple(h.shape)}")

assert z.shape == (B, T, latent_dim), \
    f"rssm z: expected (2, 8, 256), got {z.shape}"
print(f"✅  rssm z            : {tuple(z.shape)}")

assert post_logits.shape == (B, T, num_cats, cat_size), \
    f"post_logits: expected (2, 8, 16, 16), got {post_logits.shape}"
print(f"✅  post_logits       : {tuple(post_logits.shape)}")

assert prior_logits.shape == (B, T, num_cats, cat_size), \
    f"prior_logits: expected (2, 8, 16, 16), got {prior_logits.shape}"
print(f"✅  prior_logits      : {tuple(prior_logits.shape)}")

# ── image decoder ────────────────────────────────────────────────────────── #
decoder = ImageDecoder(latent_dim, hidden_dim, image_size)
pred    = decoder(h, z)
assert pred.shape == (B, T, 3, image_size, image_size), \
    f"decoder output: expected (2, 8, 3, 64, 64), got {pred.shape}"
print(f"✅  decoder output    : {tuple(pred.shape)}")

# ── mlp heads ───────────────────────────────────────────────────────────── #
reward_head = MLPHead(feat_dim, out_dim=1)
cont_head   = MLPHead(feat_dim, out_dim=1)
r_out = reward_head(h, z)
c_out = cont_head(h, z)

assert r_out.shape == (B, T, 1), \
    f"reward head: expected (2, 8, 1), got {r_out.shape}"
print(f"✅  reward head       : {tuple(r_out.shape)}")

assert c_out.shape == (B, T, 1), \
    f"continue head: expected (2, 8, 1), got {c_out.shape}"
print(f"✅  continue head     : {tuple(c_out.shape)}")

# ── actor / critic ──────────────────────────────────────────────────────── #
feat   = torch.cat([h, z], dim=-1)   # (B, T, feat_dim)
actor  = Actor(feat_dim, action_dim)
critic = Critic(feat_dim)

a_out = actor(feat)
v_out = critic(feat)

assert a_out.shape == (B, T, action_dim), \
    f"actor forward: expected (2, 8, 7), got {a_out.shape}"
print(f"✅  actor forward     : {tuple(a_out.shape)}")

assert v_out.shape == (B, T, 1), \
    f"critic forward: expected (2, 8, 1), got {v_out.shape}"
print(f"✅  critic forward    : {tuple(v_out.shape)}")

# ── losses ──────────────────────────────────────────────────────────────── #
kl  = kl_loss(post_logits, prior_logits)
assert kl.shape == torch.Size([]), \
    f"kl_loss: expected scalar, got {kl.shape}"
print(f"✅  kl_loss           : scalar {kl.item():.4f}")

rec = recon_loss(pred, images)
assert rec.shape == torch.Size([]), \
    f"recon_loss: expected scalar, got {rec.shape}"
print(f"✅  recon_loss        : scalar {rec.item():.4f}")

rewards_f = torch.randn(B, HORIZON)
values_f  = torch.randn(B, HORIZON)
dones_f   = torch.zeros(B, HORIZON)
lam_ret   = lambda_returns(rewards_f, values_f, dones_f)
assert lam_ret.shape == (B, HORIZON), \
    f"lambda_returns: expected (2, 10), got {lam_ret.shape}"
print(f"✅  lambda_returns    : {tuple(lam_ret.shape)}")

# ── imagine ─────────────────────────────────────────────────────────────── #
h0, z0 = h[:, -1], z[:, -1]
imag_h, imag_z, imag_acts = rssm.imagine(actor, h0, z0, HORIZON)

assert imag_h.shape == (B, HORIZON, hidden_dim), \
    f"imagine h: expected (2, 10, 128), got {imag_h.shape}"
print(f"✅  imagine h         : {tuple(imag_h.shape)}")

assert imag_z.shape == (B, HORIZON, latent_dim), \
    f"imagine z: expected (2, 10, 256), got {imag_z.shape}"
print(f"✅  imagine z         : {tuple(imag_z.shape)}")

assert imag_acts.shape == (B, HORIZON, action_dim), \
    f"imagine actions: expected (2, 10, 7), got {imag_acts.shape}"
print(f"✅  imagine actions   : {tuple(imag_acts.shape)}")

# ── replay buffer ───────────────────────────────────────────────────────── #
episodes = [
    {
        'images':  np.random.randint(0, 256, (30, 64, 64, 3), dtype=np.uint8),
        'actions': np.random.randn(30, 7).astype(np.float32),
        'rewards': np.random.randn(30).astype(np.float32),
        'dones':   np.zeros(30, dtype=bool),
    }
    for _ in range(5)
]
buf   = ReplayBuffer(episodes, chunk_len)
batch = buf.sample(BATCH)

assert batch['images'].shape == (BATCH, chunk_len, 64, 64, 3), \
    f"replay sample: expected (4, 24, 64, 64, 3), got {batch['images'].shape}"
print(f"✅  replay sample     : {tuple(batch['images'].shape)}")

print("\n✅  ALL TESTS PASSED")
