"""Read one feed's tracking (ATD) into numpy arrays, cached in data/cache/.

The raw file is ~200 MB of text, so it is parsed once. Positions are in metres, speed in m/s, NaN
where unseen. Row i is frame i of that feed's own video (ATD frame_id - 1).
"""

import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

CACHE_DIR = os.path.join(DATA_DIR, "cache")
FPS = 25


def parse(raw_path, n_frames, pitch):
    """(players [T,n,3], ball [T,3]) as x metres, y metres, speed m/s."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    rows = pd.read_csv(io.StringIO(text), header=None).to_numpy(np.float64)
    if not np.array_equal(rows[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame ids are not 1..{}".format(raw_path, n_frames))
    scale = np.array([pitch[0], pitch[1], 1.0])
    players = rows[:, 1:-3].reshape(n_frames, -1, 3) * scale
    return players.astype(np.float32), (rows[:, -3:] * scale).astype(np.float32)


def load(source="tactical", game=GAME):
    """dict: players [T,n,3], ball [T,3], team [n] (0 home, 1 away)."""
    cache = os.path.join(CACHE_DIR, "{}_atd.npz".format(source))
    if os.path.exists(cache):
        with np.load(cache) as cached:
            return dict(cached)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    players, ball = parse(raw_path, source_frames(source, game), GAMES[game]["pitch"])
    data = {"players": players, "ball": ball, "team": team_of_each_track(meta_path)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **data)
    return data
