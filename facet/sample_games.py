"""Metrica's open sample games: other matches, with events, used to train and tune.

Source: https://github.com/metrica-sports/sample-data (games 1 and 2). They are public, they are
not the competition match, and their labels come from Metrica's own event data. Nothing here
touches the competition match's ground truth.

    python -m facet.sample_games     downloads the files into data/external/metrica/
"""

import os
import urllib.request

import numpy as np
import pandas as pd

from facet.load import FPS
from games import DATA_DIR

FOLDER = os.path.join(DATA_DIR, "external", "metrica")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
GAMES = (1, 2)
PITCH = (105.0, 68.0)
STOPS = ("BALL OUT", "FAULT RECEIVED", "CARD")          # play stops here...
OWNING = ("SET PIECE", "PASS", "SHOT", "RECOVERY", "BALL LOST")   # ...and these name the owner

# The sample games are hand-corrected: their ball is glued to whoever has it and never lost in
# play. Ours is automatic. These settings damage the sample ball until it looks like ours. They
# were set by matching two things measured on our tracking with no labels involved: how far the
# nearest player is from the ball (median 1.65 m) and how smoothly the ball moves.
SMOOTH = 9              # frames: our tracker rounds off sharp changes of direction
WANDER = 0.8            # m: how far the tracked ball strays from the real one
WANDER_FRAMES = 25      # how slowly that error changes
HIDE_SHARE = 0.25       # of live frames lose their ball, as our tracker does in play
HIDE_SECONDS = (3, 15)


def _path(game, kind):
    return os.path.join(FOLDER, "Sample_Game_{0}_Raw{1}.csv".format(game, kind))


def download():
    os.makedirs(FOLDER, exist_ok=True)
    for game in GAMES:
        for kind in ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team"):
            if not os.path.exists(_path(game, kind)):
                name = os.path.basename(_path(game, kind))
                urllib.request.urlretrieve("{0}Sample_Game_{1}/{2}".format(URL, game, name),
                                           _path(game, kind))


def _speed(xy, half_window=12):
    """Speed in m/s over about a second, which rides over frame-to-frame jitter."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    step = xy[2 * half_window:] - xy[:-2 * half_window]
    out[half_window:-half_window] = np.hypot(step[..., 0], step[..., 1]) * FPS / (2 * half_window)
    return out


def with_speed(xy):
    """[T,n,2] positions -> [T,n,3] with speed, the layout facet.load uses."""
    return np.concatenate([xy, _speed(xy)[..., None]], axis=-1).astype(np.float32)


def load(game):
    """dict: players [T,n,3], ball_xy [T,2] (metres), team [n], labels [T] of 'H' 'A' 'D',
    period [T]."""
    download()
    sides = [pd.read_csv(_path(game, "TrackingData_{}_Team".format(side)), skiprows=3,
                         header=None).to_numpy(np.float64) for side in ("Home", "Away")]
    home, away = sides[0][:, 3:-2], sides[1][:, 3:-2]
    xy = np.concatenate([home, away], axis=1).reshape(len(home), -1, 2) * PITCH
    team = np.r_[np.zeros(home.shape[1] // 2, np.int8), np.ones(away.shape[1] // 2, np.int8)]
    period = sides[0][:, 0].astype(int)
    n = len(xy)

    events = pd.read_csv(_path(game, "EventsData")).sort_values("Start Frame", kind="stable")
    dead = np.zeros(n + 1, dtype=bool)
    owner = np.full(n + 1, -1, dtype=np.int8)
    stopped_at = 1                                       # dead until the first kick-off
    for row in events.itertuples(index=False):
        kind, sub, start, end = row[1], str(row[2]), int(row[4]), int(row[6])
        if kind == "SET PIECE" and stopped_at is not None:
            dead[stopped_at:start] = True
            stopped_at = None
        elif stopped_at is None and (kind in STOPS or (kind == "SHOT" and ("GOAL" in sub
                                                                          or "OUT" in sub))):
            stopped_at = max(end, start)
        if kind in OWNING:
            owner[start] = 0 if row[0] == "Home" else 1
    for frame in range(1, n + 1):                        # the owner keeps the ball until the next
        if owner[frame] < 0:
            owner[frame] = owner[frame - 1]
    # the end of each half is dead from its last stop onwards
    labels = np.where(dead[1:], "D", np.where(owner[1:] == 0, "H", "A"))
    return {"players": with_speed(xy), "ball_xy": sides[0][:, -2:] * PITCH, "team": team,
            "labels": labels, "period": period}


def _smooth(values, frames):
    """Mean over a centred window, ignoring NaN; NaN stays NaN."""
    ok = ~np.isnan(values)
    kernel = np.ones(frames)
    total = np.convolve(np.where(ok, values, 0.0), kernel, mode="same")
    return np.where(ok, total / np.maximum(np.convolve(ok.astype(float), kernel, mode="same"), 1),
                    np.nan)


def like_ours(game, seed):
    """A copy of a game whose ball is as rough as our tracking: rounded off, wandering, and
    missing on random stretches of live play."""
    rng = np.random.default_rng(seed)
    n = len(game["ball_xy"])
    window = np.hanning(2 * WANDER_FRAMES + 1)
    window /= np.sqrt((window ** 2).sum())
    white = rng.normal(0, WANDER, (n + 4 * WANDER_FRAMES, 2))
    wander = np.stack([np.convolve(white[:, axis], window, mode="same") for axis in (0, 1)], 1)
    ball = np.stack([_smooth(game["ball_xy"][:, axis], SMOOTH) for axis in (0, 1)], 1) \
        + wander[2 * WANDER_FRAMES:2 * WANDER_FRAMES + n]

    live = np.where(game["labels"] != "D")[0]
    hidden = 0
    while hidden < HIDE_SHARE * len(live):
        start = rng.choice(live)
        length = rng.integers(HIDE_SECONDS[0] * FPS, HIDE_SECONDS[1] * FPS)
        ball[start:start + length] = np.nan
        hidden += length
    return dict(game, ball_xy=ball)


if __name__ == "__main__":
    for number in GAMES:
        game = load(number)
        labels = game["labels"]
        starts = np.r_[True, labels[1:] != labels[:-1]]
        spells = int((starts & (labels != "D")).sum())
        print("game {}: {} frames, dead {:.1%}, {} possessions of mean {:.1f} s, ball seen "
              "{:.1%}".format(number, len(labels), (labels == "D").mean(), spells,
                              (labels != "D").sum() / spells / FPS,
                              (~np.isnan(game["ball_xy"][:, 0])).mean()))
