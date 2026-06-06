# dreamer_robot

Minimal DreamerV3-style world model for robot manipulation.

Trains on BridgeData V2 (RLDS) via Google Colab (T4 GPU).  
All shape logic is verified locally on CPU with `test_shapes.py`.

## Quick start (local CPU)

```bash
pip install -r requirements_local.txt
pip install -e .
python test_shapes.py
```

## Colab

```python
!git clone https://github.com/<you>/dreamer_robot
%cd dreamer_robot
!pip install -r requirements_colab.txt -q
!pip install -e . -q
```

## Config

| param | value |
|---|---|
| image_size | 64 |
| embed_dim | 128 |
| hidden_dim | 128 |
| num_cats × cat_size | 16 × 16 = 256 latent |
| action_dim | 7 |
| feat_dim | 384 |
| batch / chunk | 8 / 24 |
| imagine horizon | 10 |
