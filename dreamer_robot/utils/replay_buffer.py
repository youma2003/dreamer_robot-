import random
import numpy as np
import torch


class ReplayBuffer:
    """
    episodes : list of dicts with keys
        images   (T, 64, 64, 3) uint8
        actions  (T, 7)         float32
        rewards  (T,)           float32
        dones    (T,)           bool
    """

    def __init__(self, episodes, chunk_len):
        self.episodes  = episodes
        self.chunk_len = chunk_len

    def sample(self, batch_size):
        chunks = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            T  = len(ep['images'])

            if T <= self.chunk_len:
                pad   = self.chunk_len - T
                chunk = {
                    k: np.concatenate([v, np.repeat(v[-1:], pad, axis=0)], axis=0)
                    for k, v in ep.items()
                }
            else:
                start = random.randint(0, T - self.chunk_len)
                chunk = {k: v[start:start + self.chunk_len] for k, v in ep.items()}

            chunks.append(chunk)

        return {
            k: torch.from_numpy(np.stack([c[k] for c in chunks]))
            for k in chunks[0]
        }
