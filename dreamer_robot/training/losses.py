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

    return targets
