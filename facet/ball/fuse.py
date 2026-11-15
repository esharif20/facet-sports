"""One ball track from every source, with real height where two cameras allow it."""

import numpy as np
import pandas as pd

from facet import load, sync
from facet.ball import height
from games import GAME, GAMES, half_video_blocks
from starter import load_homography

AIRBORNE = 1.0         # m: above this the triangulated position replaces the projected one
OFF_PITCH = 2.0        # m: a broadcast-only ball further out than this is not trusted


def ball_track(game=GAME):
    """dict of [T] arrays on tactical frames: xy [T,2] m (under the ball), z m (NaN = unknown),
    seen, and offsets (the broadcast sync that was found)."""
    tac, bro = load.load("tactical", game), load.load("broadcast", game)
    n = len(tac["ball"])
    mapping, offsets = sync.broadcast_to_tactical(tac["ball"], bro["ball"], game)
    tac_xy = tac["ball"][:, :2].astype(np.float64)
    bro_xy = sync.onto_tactical(bro["ball"][:, :2], mapping, n)

    tac_h = load_homography.load(game)[0]
    bro_h = load_homography.load(game, source="broadcast")[0]
    in_play = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        in_play[first:last + 1] = True
    point, tac_camera, bro_camera = height.ball_3d(
        tac_xy, bro_xy, height.camera_centres(tac_h, game),
        sync.onto_tactical(height.camera_centres(bro_h, game), mapping, n), in_play)

    # a 5-frame median removes single-frame height spikes without flattening real flights
    z = pd.Series(point[:, 2]).rolling(5, center=True, min_periods=1).median().to_numpy().copy()
    z[np.isnan(point[:, 2])] = np.nan
    xy = tac_xy.copy()
    up = z > AIRBORNE
    xy[up] = point[up, :2]

    length, width = GAMES[game]["pitch"]
    only_bro = np.isnan(tac_xy[:, 0]) & ~np.isnan(bro_xy[:, 0])
    only_bro &= ((bro_xy[:, 0] > -OFF_PITCH) & (bro_xy[:, 0] < length + OFF_PITCH)
                 & (bro_xy[:, 1] > -OFF_PITCH) & (bro_xy[:, 1] < width + OFF_PITCH))
    xy[only_bro] = bro_xy[only_bro]
    return {"xy": xy, "z": z, "seen": ~np.isnan(xy[:, 0]), "offsets": offsets,
            "cameras": (tac_camera, bro_camera)}
