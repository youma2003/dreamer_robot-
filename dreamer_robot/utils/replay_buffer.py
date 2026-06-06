import random
import numpy as np
import torch


class ReplayBuffer:
    """
    episodes : list of dicts with keys
        images      (T, 64, 64, 3) uint8   numpy
        actions     (T, 7)         float32 numpy
        rewards     (T,)           float32 numpy
        dones       (T,)           bool    numpy
        instruction str   ← string keys are passed through, not concatenated
    """

    def __init__(self, episodes, chunk_len):
        self.episodes  = episodes
        self.chunk_len = chunk_len

    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_array(v) -> bool:
        return isinstance(v, np.ndarray)

    def sample(self, batch_size):
        chunks = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            T  = len(ep["images"])

            if T <= self.chunk_len:
                pad   = self.chunk_len - T
                chunk = {}
                for k, v in ep.items():
                    if self._is_array(v):
                        # pad by repeating the last frame/action/reward/done
                        chunk[k] = np.concatenate(
                            [v, np.repeat(v[-1:], pad, axis=0)], axis=0
                        )
                    else:
                        # string keys (e.g. 'instruction') — carry through as-is
                        chunk[k] = v
            else:
                start = random.randint(0, T - self.chunk_len)
                chunk = {}
                for k, v in ep.items():
                    if self._is_array(v):
                        chunk[k] = v[start : start + self.chunk_len]
                    else:
                        chunk[k] = v

            chunks.append(chunk)

        # build output: tensors for numpy keys, plain value for string keys
        result = {}
        for k in chunks[0]:
            if self._is_array(chunks[0][k]):
                result[k] = torch.from_numpy(np.stack([c[k] for c in chunks]))
            else:
                result[k] = chunks[0][k]   # e.g. instruction — take first chunk's value
        return result
