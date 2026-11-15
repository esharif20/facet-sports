"""One ball track from the two feeds, and a flag for frames where they contradict each other."""

import numpy as np

from facet import load, sync
from games import GAME, GAMES

DISPUTED = 3.0         # m: the two feeds place the ball further apart than a player can reach
OFF_PITCH = 2.0        # m: a broadcast-only ball further out than this is not trusted


def ball_track(game=GAME):
    """dict of arrays on tactical frames: xy [T,2] in metres, seen [T], disputed [T], and the
    broadcast offsets that were found.

    The tactical ball is used where it exists, the synced broadcast ball where it does not.
    Where both exist but disagree, the ball is in the air (each camera projects it to a different
    ground point) or one tracker is following something else. Either way its position cannot be
    trusted, so the frame is marked disputed.
    """
    tac, bro = load.load("tactical", game), load.load("broadcast", game)
    n = len(tac["ball"])
    mapping, offsets = sync.broadcast_to_tactical(tac["ball"], bro["ball"], game)
    xy = tac["ball"][:, :2].astype(np.float64)
    bro_xy = sync.onto_tactical(bro["ball"][:, :2], mapping, n)
    with np.errstate(invalid="ignore"):
        disputed = np.hypot(*(xy - bro_xy).T) > DISPUTED          # NaN compares False

    length, width = GAMES[game]["pitch"]
    only_bro = np.isnan(xy[:, 0]) & ~np.isnan(bro_xy[:, 0])
    only_bro &= ((bro_xy[:, 0] > -OFF_PITCH) & (bro_xy[:, 0] < length + OFF_PITCH)
                 & (bro_xy[:, 1] > -OFF_PITCH) & (bro_xy[:, 1] < width + OFF_PITCH))
    xy[only_bro] = bro_xy[only_bro]
    return {"xy": xy, "seen": ~np.isnan(xy[:, 0]), "disputed": disputed, "offsets": offsets}
