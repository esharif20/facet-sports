"""Remove ghost players: one real player tracked twice.

A ghost is born next to a real player and stays glued to them, so most of its life is spent within
a metre of one other track. A real opponent in a duel also gets that close, but has a long life
before and after, so only the track that is mostly glue is removed.
"""

import numpy as np

GLUE_DISTANCE = 1.0    # m
MIN_GLUE = 50          # frames (2 s) two tracks must stay together
MOSTLY_GLUE = 0.5      # share of a track's life spent glued for it to be the ghost


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(edges == 1)[0], np.where(edges == -1)[0]


def _life(seen):
    """Length of the unbroken track each frame belongs to, per slot [T,n]."""
    life = np.zeros(seen.shape, dtype=np.int32)
    for slot in range(seen.shape[1]):
        for start, end in zip(*_runs(seen[:, slot])):
            life[start:end, slot] = end - start
    return life


def drop_ghosts(players):
    """A copy of players [T,n,3] with ghost stretches set to NaN, and how many frames were cut."""
    xy = players[:, :, :2]
    seen = ~np.isnan(xy[:, :, 0])
    life = _life(seen)
    cleaned, cut = players.copy(), 0
    n = players.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            gap = xy[:, i] - xy[:, j]
            close = np.hypot(gap[:, 0], gap[:, 1]) < GLUE_DISTANCE      # NaN compares False
            for start, end in zip(*_runs(close)):
                if end - start < MIN_GLUE:
                    continue
                ghost = i if life[start, i] <= life[start, j] else j
                if (end - start) / life[start, ghost] >= MOSTLY_GLUE:
                    cleaned[start:end, ghost] = np.nan
                    cut += end - start
    return cleaned, cut
