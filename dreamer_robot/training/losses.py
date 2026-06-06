"""
training/losses.py — loss functions for DreamerV3 robot world model
====================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COLAB CELL 9 — PHASE B REPLACEMENT  (actor-critic train step)
  Paste the block below in place of your existing Phase B section.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Phase B: Actor-Critic on imagined rollouts ───────────────────── #
# Seeds imagination from the last latent state of the observed chunk.
# Five stability fixes applied here (see losses.py for rationale):
#   1. lambda_returns normalises targets  → prevents scale explosion
#   2. log_prob clamped to [-10, 0]       → prevents -inf × big target
#   3. critic scaled by 0.5               → standard value-loss weight
#   4. grad clip = 1.0 (was 100.0)        → tight leash on actor grads
#   5. log_std clamped in Actor.distribution() (in actor_critic.py)

import torch.nn.functional as F   # already imported in cell 1

h0 = h[:, -1].detach()            # (B, hidden_dim) — seed from observed
z0 = z[:, -1].detach()            # (B, latent_dim)

with torch.no_grad():
    imag_h, imag_z, imag_acts = rssm.imagine(actor, h0, z0, imagine_horizon)

imag_feat = torch.cat([imag_h, imag_z], dim=-1)    # (B, H, feat_dim)

# predicted rewards and continue flags along imagination
with torch.no_grad():
    imag_r = reward_head(imag_h, imag_z).squeeze(-1)           # (B, H)
    imag_c = cont_head(imag_h, imag_z).squeeze(-1).sigmoid()   # (B, H)
    imag_done = (1.0 - imag_c)                                 # (B, H)

imag_v = critic(imag_feat).squeeze(-1)                         # (B, H)

# lambda_returns normalises output to zero-mean, unit-std (fix #1)
targets = lambda_returns(
    imag_r, imag_v.detach(), imag_done.detach()
)                                                              # (B, H)

# actor loss — log_prob clamped before multiplying returns (fix #2)
dist  = actor.distribution(imag_feat[:, :-1])                  # (B, H-1)
log_p = dist.log_prob(imag_acts[:, :-1]).sum(-1)               # (B, H-1)
log_p = log_p.clamp(-10, 0)                                    # fix #2
l_actor  = -(log_p * targets[:, :-1].detach()).mean()

# critic loss — 0.5 scaling (fix #5)
l_critic = 0.5 * F.mse_loss(imag_v, targets.detach())         # fix #3

l_ac = l_actor + l_critic
optimizer_ac.zero_grad()
l_ac.backward()
torch.nn.utils.clip_grad_norm_(                                # fix #4
    list(actor.parameters()) + list(critic.parameters()),
    max_norm=1.0,                                              # was 100.0
)
optimizer_ac.step()

# logging
if step % log_every == 0:
    print(f"step {step:6d} | "
          f"recon {l_recon.item():.4f} | "
          f"kl {l_kl.item():.4f} | "
          f"actor {l_actor.item():.4f} | "
          f"critic {l_critic.item():.4f}")

# ── end Phase B replacement ──────────────────────────────────────── #
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn.functional as F


def symlog(x):
    return x.sign() * (x.abs() + 1).log()


def symexp(x):
    return x.sign() * (x.abs().exp() - 1)


def kl_loss(post_logits, prior_logits, free_nats=1.0):
    """
    post_logits / prior_logits : (B, T, num_cats, cat_size)
    Returns scalar.
    """
    post = F.softmax(post_logits, dim=-1)
    kl   = (post * (F.log_softmax(post_logits,  dim=-1)
                  - F.log_softmax(prior_logits, dim=-1))).sum(-1).sum(-1)
    return kl.clamp(min=free_nats).mean()


def recon_loss(pred, target):
    """
    pred   : (B, T, 3, H, W)  raw logits
    target : (B, T, H, W, 3)  uint8
    """
    target_norm = target.float() / 255.0
    target_norm = target_norm.permute(0, 1, 4, 2, 3)   # → (B, T, 3, H, W)
    return F.mse_loss(pred.sigmoid(), target_norm)


def lambda_returns(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards / values / dones : (B, H) each
    Returns targets           : (B, H)
    """
    B, H    = rewards.shape
    targets = torch.zeros_like(rewards)
    last    = values[:, -1]

    for t in reversed(range(H)):
        cont           = 1.0 - dones[:, t].float()
        td             = rewards[:, t] + gamma * cont * last
        targets[:, t]  = td + gamma * lam * cont * (last - values[:, t])
        last           = targets[:, t]

    # Normalise to zero mean / unit std so real-robot reward variance
    # doesn't blow up the actor loss when targets are multiplied by log_prob.
    targets = (targets - targets.mean()) / (targets.std() + 1e-8)
    return targets
