import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, action_dim, num_cats=16, cat_size=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_cats   = num_cats
        self.cat_size   = cat_size
        self.latent_dim = num_cats * cat_size

        self.gru = nn.GRUCell(self.latent_dim + action_dim, hidden_dim)

        self.repr_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_cats * cat_size),
        )
        self.trans_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_cats * cat_size),
        )

    # ------------------------------------------------------------------ #

    def initial_state(self, B, device):
        h = torch.zeros(B, self.hidden_dim, device=device)
        z = torch.zeros(B, self.latent_dim,  device=device)
        return h, z

    def _straight_through(self, logits):
        # logits: (B, num_cats, cat_size)
        probs   = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(probs.argmax(-1), num_classes=self.cat_size).float()
        z_st    = (one_hot - probs).detach() + probs
        return z_st.flatten(1)                                 # (B, latent_dim)

    # ------------------------------------------------------------------ #

    def step(self, h, z, action, obs_embed=None):
        """
        h         : (B, hidden_dim)
        z         : (B, latent_dim)
        action    : (B, action_dim)   — the action taken at the *previous* timestep
        obs_embed : (B, embed_dim) or None   (None → imagination / prior mode)
        Returns   : h_new, z_new, post_logits, prior_logits  all (B, ...)
        """
        h_new = self.gru(torch.cat([z, action], dim=-1), h)

        prior_logits = self.trans_net(h_new).reshape(-1, self.num_cats, self.cat_size)

        if obs_embed is not None:
            post_logits = self.repr_net(
                torch.cat([h_new, obs_embed], dim=-1)
            ).reshape(-1, self.num_cats, self.cat_size)
        else:
            post_logits = prior_logits

        z_new = self._straight_through(post_logits)
        return h_new, z_new, post_logits, prior_logits

    # ------------------------------------------------------------------ #

    def observe(self, obs_embeds, actions):
        """
        obs_embeds : (B, T, embed_dim)
        actions    : (B, T, action_dim)  — actions[t] is a_t (taken after o_t)
        Returns    : h, z, post_logits, prior_logits  all (B, T, ...)
        """
        B, T, _ = obs_embeds.shape
        device  = obs_embeds.device
        h, z    = self.initial_state(B, device)

        hs, zs, posts, priors = [], [], [], []
        for t in range(T):
            a_prev = (actions[:, t - 1] if t > 0
                      else torch.zeros(B, self.action_dim, device=device))
            h, z, post, prior = self.step(h, z, a_prev, obs_embeds[:, t])
            hs.append(h);  zs.append(z)
            posts.append(post); priors.append(prior)

        return (
            torch.stack(hs,     dim=1),   # (B, T, hidden_dim)
            torch.stack(zs,     dim=1),   # (B, T, latent_dim)
            torch.stack(posts,  dim=1),   # (B, T, num_cats, cat_size)
            torch.stack(priors, dim=1),   # (B, T, num_cats, cat_size)
        )

    # ------------------------------------------------------------------ #

    def imagine(self, actor, h0, z0, horizon):
        """
        actor   : Actor instance  (forward(feat) → tanh action)
        h0, z0  : (B, hidden_dim / latent_dim) — starting states
        Returns : h, z, actions  all (B, horizon, ...)
        """
        h, z = h0, z0
        hs, zs, acts = [], [], []
        for _ in range(horizon):
            feat   = torch.cat([h, z], dim=-1)
            action = actor(feat)
            h, z, _, _ = self.step(h, z, action)
            hs.append(h); zs.append(z); acts.append(action)

        return (
            torch.stack(hs,   dim=1),   # (B, H, hidden_dim)
            torch.stack(zs,   dim=1),   # (B, H, latent_dim)
            torch.stack(acts, dim=1),   # (B, H, action_dim)
        )
