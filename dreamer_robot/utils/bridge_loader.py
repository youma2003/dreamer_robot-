"""
utils/bridge_loader.py
======================
Download and parse real robot manipulation episodes for DreamerV3 training.

Priority order:
  1. pollen-robotics/pick_and_place_bottle  (86 real episodes, ~350 MB)
     LeRobot v2.1 — per-episode parquet + per-episode AV1 video
  2. nvidia/BridgeData2_LeRobot_v3          (50 k episodes, 7-DOF perfect)
     LeRobot v3.0 — per-chunk parquet + per-chunk AV1 video
  3. Synthetic random-walk fallback         (works everywhere, no download)

Output format per episode:
  {
    'images'     : np.ndarray (T, 64, 64, 3)  uint8
    'actions'    : np.ndarray (T, 7)          float32
    'rewards'    : np.ndarray (T,)            float32  (sparse: 1.0 on last step)
    'dones'      : np.ndarray (T,)            bool
    'instruction': str
  }

Usage:
  python -m dreamer_robot.utils.bridge_loader          # downloads real data
  python -m dreamer_robot.utils.bridge_loader --synth  # synthetic only
"""

import io, os, json, subprocess, sys
from pathlib import Path

import numpy as np
import requests

try:
    from tqdm import tqdm as _tqdm
    def _bar(it, **kw):
        return _tqdm(it, **kw)
except ImportError:
    def _bar(it, **kw):
        return it


# ════════════════════════════════════════════════════════════════════════ #
#  Dataset constants                                                        #
# ════════════════════════════════════════════════════════════════════════ #

POLLEN_REPO   = "pollen-robotics/pick_and_place_bottle"
POLLEN_N_EP   = 86
POLLEN_CAMERA = "observation.images.teleop_right"
POLLEN_TASK   = "pick and place bottle"

BRIDGE_REPO   = "nvidia/BridgeData2_LeRobot_v3"
BRIDGE_CAMERA = "observation.images.image_0"


# ════════════════════════════════════════════════════════════════════════ #
#  Low-level helpers                                                        #
# ════════════════════════════════════════════════════════════════════════ #

def _hf_url(repo_id: str, rel_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}"


def _download(url: str, dest: Path, label: str = "") -> bool:
    """
    Stream-download url → dest with optional progress bar.
    Returns True on success, False on HTTP 404 or any error.
    Skips download if dest already exists.
    """
    if dest.exists():
        return True
    try:
        r = requests.get(url, stream=True, timeout=60,
                         headers={"User-Agent": "dreamer-robot/0.1"})
        if r.status_code == 404:
            return False
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            if total:
                bar = _bar(total=total, unit="B", unit_scale=True,
                           desc=label or dest.name, leave=False)
                for chunk in r.iter_content(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
                bar.close()
            else:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception as exc:
        print(f"    ⚠  {url[:70]}  →  {exc}")
        return False


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _decode_video(video_path: Path, image_size: int = 64) -> np.ndarray:
    """
    Decode every frame of an MP4/AV1 file via ffmpeg raw pipe.
    Returns (T, image_size, image_size, 3) uint8.

    Works with any codec ffmpeg supports (including AV1 in Colab).
    No extra Python libraries needed beyond subprocess + numpy.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vf", f"scale={image_size}:{image_size}",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg failed on {video_path.name}: "
            f"{proc.stderr.decode(errors='replace')[:200]}"
        )
    frame_bytes = image_size * image_size * 3
    n = len(proc.stdout) // frame_bytes
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    return raw[: n * frame_bytes].reshape(n, image_size, image_size, 3).copy()


def _col_to_array(series, dtype=np.float32) -> np.ndarray:
    """
    Safely convert a parquet list-column (pandas Series of lists / arrays)
    into a 2-D numpy array.  Handles both PyArrow-backed and native pandas.
    """
    first = series.iloc[0]
    if isinstance(first, (list, np.ndarray)):
        return np.array(series.tolist(), dtype=dtype)
    # scalar column — should not happen for action, but be safe
    return series.to_numpy(dtype=dtype)[:, None]


def _pack_episode(frames: np.ndarray, actions: np.ndarray,
                  instruction: str = "") -> dict:
    """Build an episode dict with sparse reward (1.0 on last step)."""
    T = min(len(frames), len(actions))
    rewards        = np.zeros(T, dtype=np.float32)
    rewards[-1]    = 1.0
    dones          = np.zeros(T, dtype=bool)
    dones[-1]      = True
    return {
        "images":      frames[:T].copy(),
        "actions":     actions[:T].copy(),
        "rewards":     rewards,
        "dones":       dones,
        "instruction": instruction,
    }


# ════════════════════════════════════════════════════════════════════════ #
#  Source 1 — pollen-robotics/pick_and_place_bottle                        #
# ════════════════════════════════════════════════════════════════════════ #
#
#  File layout (per episode):
#    data/chunk-000/episode_XXXXXX.parquet
#    videos/chunk-000/<CAMERA>/episode_XXXXXX.mp4
#
#  Action shape: 8-DOF  →  we trim to 7 (drop the 8th redundant gripper dim)
# ════════════════════════════════════════════════════════════════════════ #

def _load_pollen(max_ep: int, cache: Path, image_size: int) -> list:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow not installed — skipping pollen source")
        return []

    n   = min(max_ep, POLLEN_N_EP)
    eps = []
    print(f"\n  [{POLLEN_REPO}]  downloading {n}/{POLLEN_N_EP} episodes ...")

    for i in _bar(range(n), desc="  pollen", unit="ep"):
        tag      = f"episode_{i:06d}"
        pq_path  = cache / "pollen/data/chunk-000" / f"{tag}.parquet"
        vid_path = cache / f"pollen/videos/chunk-000/{POLLEN_CAMERA}" / f"{tag}.mp4"

        ok_pq  = _download(_hf_url(POLLEN_REPO, f"data/chunk-000/{tag}.parquet"),
                           pq_path, f"pq {i:03d}")
        ok_vid = _download(
            _hf_url(POLLEN_REPO, f"videos/chunk-000/{POLLEN_CAMERA}/{tag}.mp4"),
            vid_path, f"vid {i:03d}",
        )
        if not (ok_pq and ok_vid):
            continue

        try:
            df      = pq.read_table(pq_path).to_pandas()
            actions = _col_to_array(df["action"])[:, :7]   # 8-DOF → 7-DOF
            frames  = _decode_video(vid_path, image_size)
            eps.append(_pack_episode(frames, actions, POLLEN_TASK))
        except Exception as exc:
            print(f"    episode {i:03d} error: {exc}")

    print(f"  [{POLLEN_REPO}]  ✅  {len(eps)} episodes loaded")
    return eps


# ════════════════════════════════════════════════════════════════════════ #
#  Source 2 — nvidia/BridgeData2_LeRobot_v3                               #
# ════════════════════════════════════════════════════════════════════════ #
#
#  File layout (per chunk × file):
#    data/chunk-CCC/file-FFF.parquet          ← frame rows, 7-DOF action
#    videos/<CAMERA>/chunk-CCC/file-FFF.mp4   ← video, frame k = parquet row k
#    meta/tasks.parquet                        ← task_index → instruction
#
#  Episode grouping: parquet column 'episode_index'.
#  Video frame positions map 1-to-1 with parquet rows (within each file).
# ════════════════════════════════════════════════════════════════════════ #

def _load_nvidia_bridge(max_ep: int, cache: Path, image_size: int) -> list:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow not installed — skipping nvidia bridge source")
        return []

    root     = cache / "nvidia_bridge"
    eps      = []
    task_map: dict = {}

    # -- tasks table --
    tasks_path = root / "meta/tasks.parquet"
    if _download(_hf_url(BRIDGE_REPO, "meta/tasks.parquet"), tasks_path, "tasks"):
        try:
            tt = pq.read_table(tasks_path).to_pandas()
            task_map = dict(zip(tt["task_index"].astype(int), tt["task"]))
        except Exception:
            pass

    print(f"\n  [{BRIDGE_REPO}]  downloading chunk by chunk (need {max_ep} episodes) ...")

    CHUNK_LIMIT = 10   # download at most 10 chunks (~10k episodes)
    for chunk in range(CHUNK_LIMIT):
        if len(eps) >= max_ep:
            break
        chunk_s  = f"chunk-{chunk:03d}"
        file_num = 0

        while len(eps) < max_ep:
            file_s    = f"file-{file_num:03d}"
            pq_path   = root / "data" / chunk_s / f"{file_s}.parquet"
            vid_path  = root / "videos" / BRIDGE_CAMERA / chunk_s / f"{file_s}.mp4"

            pq_url  = _hf_url(BRIDGE_REPO, f"data/{chunk_s}/{file_s}.parquet")
            vid_url = _hf_url(BRIDGE_REPO,
                              f"videos/{BRIDGE_CAMERA}/{chunk_s}/{file_s}.mp4")

            if not _download(pq_url, pq_path, f"data {chunk_s}/{file_s}"):
                break   # no more files in this chunk
            if not _download(vid_url, vid_path, f"vid  {chunk_s}/{file_s}"):
                break

            try:
                df         = pq.read_table(pq_path).to_pandas()
                all_frames = _decode_video(vid_path, image_size)

                for ep_idx, ep_df in df.groupby("episode_index"):
                    if len(eps) >= max_ep:
                        break
                    # ep_df.index.values are the 0-based row positions
                    # in this parquet file = the 0-based video frame indices
                    positions = ep_df.index.values
                    if len(positions) == 0 or int(positions.max()) >= len(all_frames):
                        continue

                    frames  = all_frames[positions]
                    actions = _col_to_array(ep_df["action"])   # (T, 7)
                    t_idx   = int(ep_df["task_index"].iloc[0])
                    inst    = task_map.get(t_idx, "robot pick and place")
                    eps.append(_pack_episode(frames, actions, inst))

            except Exception as exc:
                print(f"    {chunk_s}/{file_s} error: {exc}")

            file_num += 1

    print(f"  [{BRIDGE_REPO}]  ✅  {len(eps)} episodes loaded")
    return eps


# ════════════════════════════════════════════════════════════════════════ #
#  Source 3 — synthetic fallback                                           #
# ════════════════════════════════════════════════════════════════════════ #

def _make_synthetic(n: int = 50, image_size: int = 64) -> list:
    """
    Generate n random-walk episodes that pass all shape/dtype checks.
    Uses numpy only — no internet, no GPU, no ffmpeg required.
    """
    print(f"\n  [synthetic]  generating {n} placeholder episodes ...")
    rng = np.random.default_rng(0)
    eps = []
    for _ in range(n):
        T       = int(rng.integers(20, 50))
        images  = rng.integers(0, 256, (T, image_size, image_size, 3),
                               dtype=np.uint8)
        deltas  = rng.normal(0, 0.05, (T, 7)).astype(np.float32)
        actions = np.clip(np.cumsum(deltas, axis=0), -1.0, 1.0)
        eps.append(_pack_episode(images, actions, "synthetic pick and place"))
    print(f"  [synthetic]  ✅  {n} episodes generated")
    return eps


# ════════════════════════════════════════════════════════════════════════ #
#  Public API                                                              #
# ════════════════════════════════════════════════════════════════════════ #

def load_bridge_data(
    max_episodes: int  = 500,
    cache_dir:    str  = ".cache/bridge_data",
    image_size:   int  = 64,
    force_synthetic: bool = False,
) -> list:
    """
    Download and return up to max_episodes real robot manipulation episodes.

    Parameters
    ----------
    max_episodes     : How many episodes to return (at most).
    cache_dir        : Local directory to cache downloads.
    image_size       : Resize frames to (image_size × image_size).
    force_synthetic  : Skip all downloads and return random data.
                       Also activated by env var DREAMER_SYNTHETIC=1.

    Returns
    -------
    List of dicts; each dict has keys:
      images (T,H,W,3) uint8  |  actions (T,7) float32
      rewards (T,) float32    |  dones (T,) bool
      instruction str
    """
    cache = Path(cache_dir)

    if force_synthetic or os.environ.get("DREAMER_SYNTHETIC"):
        return _make_synthetic(min(max_episodes, 100), image_size)

    if not _has_ffmpeg():
        print("⚠  ffmpeg not found — falling back to synthetic data.")
        print("   In Google Colab, ffmpeg is pre-installed.")
        print("   Locally: install ffmpeg (brew/apt/choco install ffmpeg).")
        print("   Or set DREAMER_SYNTHETIC=1 to suppress this warning.")
        return _make_synthetic(min(max_episodes, 50), image_size)

    episodes: list = []

    # ── source 1 ────────────────────────────────────────────────────── #
    try:
        want  = min(max_episodes, POLLEN_N_EP)
        peps  = _load_pollen(want, cache, image_size)
        episodes.extend(peps)
    except Exception as exc:
        print(f"  Pollen source exception: {exc}")

    # ── source 2 (if we still need more) ────────────────────────────── #
    if len(episodes) < max_episodes:
        need = max_episodes - len(episodes)
        try:
            beps = _load_nvidia_bridge(need, cache, image_size)
            episodes.extend(beps)
        except Exception as exc:
            print(f"  BridgeData2 source exception: {exc}")

    # ── fallback ─────────────────────────────────────────────────────── #
    if not episodes:
        print("\n⚠  All real-data sources failed — using synthetic data.")
        episodes = _make_synthetic(min(max_episodes, 50), image_size)

    return episodes[:max_episodes]


# ════════════════════════════════════════════════════════════════════════ #
#  CLI self-test                                                           #
# ════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    force_synth = "--synth" in sys.argv or "--synthetic" in sys.argv
    n_test      = 5 if not force_synth else 10

    print(f"bridge_loader self-test  (force_synthetic={force_synth})\n")

    episodes = load_bridge_data(
        max_episodes     = n_test,
        force_synthetic  = force_synth,
    )

    # ── required output lines ─────────────────────────────────────────── #
    print(f"\n✅ Downloaded {len(episodes)} episodes")

    ep = episodes[0]
    T  = ep["images"].shape[0]
    print(f"✅ Sample shape: images {ep['images'].shape} actions {ep['actions'].shape}")
    print(f"✅ Sample instruction: \"{ep['instruction']}\"")

    # ── assertions ────────────────────────────────────────────────────── #
    assert len(episodes) > 0,                            "no episodes"
    assert ep["images"].dtype  == np.uint8,             f"images dtype {ep['images'].dtype}"
    assert ep["actions"].dtype == np.float32,           f"actions dtype {ep['actions'].dtype}"
    assert ep["images"].shape[1:]  == (64, 64, 3),      f"images shape {ep['images'].shape}"
    assert ep["actions"].shape[1]  == 7,                f"actions dim {ep['actions'].shape}"
    assert ep["rewards"].shape     == (T,),             f"rewards shape {ep['rewards'].shape}"
    assert ep["dones"].shape       == (T,),             f"dones shape {ep['dones'].shape}"
    assert ep["dones"][-1]         == True,              "last done not True"
    assert ep["rewards"][-1]       == 1.0,              f"last reward {ep['rewards'][-1]}"
    assert isinstance(ep["instruction"], str),           "instruction not str"

    print("✅ All assertions passed")
