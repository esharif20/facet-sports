"""Evidence for home vs away possession, as a per-frame log-likelihood ratio."""

import numpy as np

CONTROL_RADIUS = 2.5   # m: a player this close to the ball can be playing it
MAX_GAP = 3.0          # m: cap on the distance difference, so one far-off frame cannot dominate
LOOSE_WEIGHT = 0.2     # evidence weight when nobody is within CONTROL_RADIUS (ball in flight)


def team_evidence(f):
    """log P(frame | H) - log P(frame | A). Positive favours home; 0 where the ball is unseen."""
    diff = np.clip(f["near_away"] - f["near_home"], -MAX_GAP, MAX_GAP)
    close = np.fmin(f["near_home"], f["near_away"]) < CONTROL_RADIUS
    out = np.where(close, 1.0, LOOSE_WEIGHT) * diff
    return np.nan_to_num(np.where(f["ball_seen"], out, 0.0))
