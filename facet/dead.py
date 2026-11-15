"""Evidence that the ball is dead, as a per-frame log-likelihood ratio (dead vs alive)."""

import numpy as np

OUT_OF_PLAY = 4.0        # ball seen beyond the lines
WALKING_SPEED = 1.1      # m/s: mean player speed below this looks like a stoppage
SPEED_WEIGHT = 1.0
MISSING = 0.3            # ball unseen: mildly more likely dead
BALL_MOVING = 3.0        # m/s: ball seen moving inside the pitch
MOVING_ALIVE = 1.0


def dead_evidence(f):
    """log P(frame | dead) - log P(frame | alive)."""
    speed = np.clip(SPEED_WEIGHT * (WALKING_SPEED - np.nan_to_num(f["player_speed"])), -3, 3)
    moving = f["ball_seen"] & ~f["ball_out"] & (np.nan_to_num(f["ball_speed"]) > BALL_MOVING)
    return (speed
            + np.where(f["ball_seen"] & f["ball_out"], OUT_OF_PLAY, 0.0)
            + np.where(f["ball_seen"], 0.0, MISSING)
            - np.where(moving, MOVING_ALIVE, 0.0))
