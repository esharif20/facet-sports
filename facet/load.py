"""Read the tactical ATD once into numpy arrays and cache them next to the data.

The raw file is ~200 MB of text; parsing it takes about a minute, loading the cache well under a
second. Everything is in metres and indexed by video frame (ATD frame_id - 1).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

N_SLOTS = 46
VALUES_PER_LINE = 1 + N_SLOTS * 3 + 3  # frame_id, (x, y, speed) per slot, ball (x, y, speed)
CACHE = os.path.join(DATA_DIR, "cache", "tactical_atd.npz")


def parse(raw_path, n_frames, pitch):
    """(players [T,46,3], ball [T,3]) as x metres, y metres, speed m/s. NaN where unseen."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",").replace("\n", ",")
    values = np.array(text.rstrip(",").split(","), dtype=np.float64)
    if values.size != n_frames * VALUES_PER_LINE:
        raise ValueError("{}: expected {} values, got {}".format(
            raw_path, n_frames * VALUES_PER_LINE, values.size))
    rows = values.reshape(n_frames, VALUES_PER_LINE)
    if not np.array_equal(rows[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame ids are not 1..{}".format(raw_path, n_frames))

    scale = np.array([pitch[0], pitch[1], 1.0])
    players = rows[:, 1:1 + N_SLOTS * 3].reshape(n_frames, N_SLOTS, 3) * scale
    ball = rows[:, 1 + N_SLOTS * 3:] * scale
    return players, ball


def load(game=GAME):
    """dict with players [T,46,3], ball [T,3], team [46] (0 home, 1 away). Builds the cache once."""
    if os.path.exists(CACHE):
        with np.load(CACHE) as cached:
            return dict(cached)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    players, ball = parse(raw_path, source_frames("tactical", game), GAMES[game]["pitch"])
    data = {"players": players.astype(np.float32), "ball": ball.astype(np.float32),
            "team": team_of_each_track(meta_path)}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **data)
    return data


if __name__ == "__main__":
    d = load()
    seen = ~np.isnan(d["ball"][:, 0])
    print("frames {}  ball seen {:.1%}  mean tracked players {:.1f}".format(
        len(seen), seen.mean(), (~np.isnan(d["players"][:, :, 0])).sum(1).mean()))
