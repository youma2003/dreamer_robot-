# PASTE THIS AS CELL 4 IN COLAB — replaces synthetic data
# ─────────────────────────────────────────────────────────────────────────
# Cell 4 · Real BridgeData / Pollen pick-and-place data pipeline
#
# What this cell does:
#   1. Installs / verifies all required packages
#   2. Clones your dreamer_robot repo (edit REPO_URL below)
#   3. Downloads real robot episodes into .cache/bridge_data/
#      Priority: pollen-robotics/pick_and_place_bottle (86 ep, ~350 MB)
#                nvidia/BridgeData2_LeRobot_v3 (top-up to MAX_EPISODES)
#   4. Builds the ReplayBuffer used by the rest of the notebook
#
# Expected output:
#   ✅ Downloaded N episodes
#   ✅ Sample shape: images (T, 64, 64, 3) actions (T, 7)
#   ✅ Sample instruction: "..."
# ─────────────────────────────────────────────────────────────────────────

# ── 0 · config (edit these) ──────────────────────────────────────────── #
REPO_URL     = "https://github.com/youma2003/dreamer_robot-.git"
MAX_EPISODES = 500      # total episodes to load (pollen + bridge top-up)
IMAGE_SIZE   = 64       # must match your model config
CACHE_DIR    = "/content/bridge_cache"

# ── 1 · install packages ─────────────────────────────────────────────── #
import subprocess, sys

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pip("pyarrow", "requests", "tqdm")
# Note: ffmpeg + torch + torchvision are pre-installed in Colab

# ── 2 · clone repo (skip if already present) ─────────────────────────── #
import os
from pathlib import Path

REPO_DIR = Path("/content/dreamer_robot")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
                   check=True)
    print(f"✅ Cloned {REPO_URL}")
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
                   capture_output=True)
    print("✅ Repo already present (pulled latest)")

# add to sys.path so imports work
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# ── 3 · download episodes ─────────────────────────────────────────────── #
from dreamer_robot.utils.bridge_loader import load_bridge_data

print(f"\nLoading up to {MAX_EPISODES} episodes (cache: {CACHE_DIR}) ...")
episodes = load_bridge_data(
    max_episodes = MAX_EPISODES,
    cache_dir    = CACHE_DIR,
    image_size   = IMAGE_SIZE,
)

# ── 4 · required status lines ────────────────────────────────────────── #
import numpy as np
print(f"\n✅ Downloaded {len(episodes)} episodes")

ep = episodes[0]
print(f"✅ Sample shape: images {ep['images'].shape} actions {ep['actions'].shape}")
print(f"✅ Sample instruction: \"{ep['instruction']}\"")

# quick sanity check
assert ep["images"].dtype  == np.uint8,   "images must be uint8"
assert ep["actions"].shape[1] == 7,       "actions must be 7-DOF"
assert ep["rewards"][-1]   == 1.0,        "last reward must be 1.0"

# ── 5 · build ReplayBuffer (used by the rest of the notebook) ──────── #
from dreamer_robot.utils.replay_buffer import ReplayBuffer

CHUNK_LEN  = 24    # must match your training config
replay     = ReplayBuffer(episodes, chunk_len=CHUNK_LEN)

# verify a batch
batch = replay.sample(batch_size=8)
print(f"\n✅ ReplayBuffer ready:")
print(f"   images  {tuple(batch['images'].shape)}")
print(f"   actions {tuple(batch['actions'].shape)}")
print(f"   rewards {tuple(batch['rewards'].shape)}")
print(f"\n{'─'*60}")
print(f"  Total episodes : {len(episodes)}")
print(f"  Episode lengths: min={min(len(e['images']) for e in episodes)}"
      f"  max={max(len(e['images']) for e in episodes)}")
print(f"  Unique tasks   : {len(set(e['instruction'] for e in episodes))}")
print(f"{'─'*60}")
