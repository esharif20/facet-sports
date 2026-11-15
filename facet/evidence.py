"""Per-frame evidence for the decoder, as log-likelihood ratios.

team:  positive favours home, negative favours away, 0 says nothing.
dead:  positive favours a dead ball, negative favours live play.
"""

import warnings

import numpy as np

from facet.load import FPS
from games import GAME, GAMES, half_video_blocks
from starter import load_smart_tagging

# Control: a player close to the ball and moving with it (Link & Hoernig 2017).
CONTROL_RADIUS = 2.0     # m
MAX_REL_SPEED = 5.0      # m/s between ball and player; a ball flying past is faster than this
CARRY = 0.01             # loose ball: a faint vote for the team that last controlled it

# Dead ball.
WALKING = 1.4            # m/s mean player speed: slower looks like a stoppage, faster like play
SPEED_WEIGHT = 2.0
MISSING = 1.0            # the tracker loses the ball mostly when it leaves play
OUT = 3.0                # ball seen beyond the lines
OUT_MARGIN = 1.5         # m
STILL = 1.5              # ball seen but not moving
STILL_DISTANCE = 0.5     # m moved in one second
MOVING = 3.0             # m/s

# Smart Tagging phases that name the attacking team.
ATTACKING_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "SET PIECES")
TAG_WEIGHT = 0.03


def velocity(xy, half_window=3):
    """Velocity [.., 2] in m/s by central difference; NaN where either end is unseen."""
    out = np.full(xy.shape, np.nan)
    out[half_window:-half_window] = (xy[2 * half_window:] - xy[:-2 * half_window]) \
        * (FPS / (2 * half_window))
    return out


def team_evidence(ball, players, team):
    """Who controls the ball. ball: dict from fuse.ball_track; players [T,n,3]; team [n]."""
    xy = players[:, :, :2].astype(np.float64)
    offset = xy - ball["xy"][:, None, :]
    dist = np.hypot(offset[:, :, 0], offset[:, :, 1])
    rel = velocity(xy) - velocity(ball["xy"])[:, None, :]
    rel_speed = np.hypot(rel[:, :, 0], rel[:, :, 1])
    # an unknown speed (track edge) does not rule a player out
    dist = np.where(np.isnan(rel_speed) | (rel_speed < MAX_REL_SPEED), dist, np.nan)
    with warnings.catch_warnings():          # a frame with no candidate for a team is NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        home = np.nanmin(np.where(team == 0, dist, np.nan), axis=1)
        away = np.nanmin(np.where(team == 1, dist, np.nan), axis=1)
    home = np.where(np.isnan(home) | (home > CONTROL_RADIUS), np.inf, home)
    away = np.where(np.isnan(away) | (away > CONTROL_RADIUS), np.inf, away)
    # both close means contested, which says little; one close says a lot
    llr = np.clip(np.where(np.isinf(home) & np.isinf(away), 0.0,
                           np.nan_to_num(away, posinf=99.0) - np.nan_to_num(home, posinf=99.0)),
                  -1.0, 1.0)
    llr[~ball["seen"] | ball["disputed"]] = 0.0

    # contested or loose: possession stays with the last team, so lean faintly their way
    last = np.sign(llr)
    for t in range(1, len(last)):
        if last[t] == 0:
            last[t] = last[t - 1]
    return np.where(llr == 0, CARRY * last, llr)


def player_speed(players, seconds=2.0):
    """Mean speed of everyone tracked, smoothed [T]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nan_to_num(np.nanmean(players[:, :, 2], axis=1))
    width = int(seconds * FPS) | 1
    return np.convolve(speed, np.ones(width) / width, mode="same")


def dead_evidence(ball, players, game=GAME):
    length, width = GAMES[game]["pitch"]
    x, y, seen = ball["xy"][:, 0], ball["xy"][:, 1], ball["seen"]
    with np.errstate(invalid="ignore"):
        out = seen & ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN)
                      | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))
        half = FPS // 2
        moved = np.full(len(x), np.nan)
        moved[half:-half] = np.hypot(x[2 * half:] - x[:-2 * half], y[2 * half:] - y[:-2 * half])
        still = seen & (moved < STILL_DISTANCE)
        speed = np.hypot(*velocity(ball["xy"]).T)
        moving = seen & ~out & (speed > MOVING)
    return (np.clip(SPEED_WEIGHT * (WALKING - player_speed(players)), -2.0, 2.0)
            + np.where(seen, -MISSING, MISSING)
            + np.where(out, OUT, 0.0)
            + np.where(still, STILL, 0.0)
            - np.where(moving, 1.0, 0.0))


def before_kickoff(n_frames, game=GAME):
    """Frames of each half before its kick-off tag starts: warm-up, never play [T] bool."""
    mask = np.zeros(n_frames, dtype=bool)
    tags = [t for t in load_smart_tagging.load(game) if t["tags"].get("Type") == "Kick Off (Start)"]
    for _, first, last in half_video_blocks(game):
        for tag in tags:
            if first < tag["start_frame"] <= last:
                mask[first:tag["start_frame"]] = True
    return mask


def tag_prior(n_frames, home_name, game=GAME):
    """A weak vote for the team Smart Tagging says is attacking [T]."""
    prior = np.zeros(n_frames)
    for tag in load_smart_tagging.load(game):
        who = tag["tags"].get("Team", "N/A")
        if tag["code"] in ATTACKING_CODES and who != "N/A":
            sign = 1.0 if who == home_name else -1.0
            prior[max(tag["start_frame"], 0):tag["end_frame"] + 1] += sign * TAG_WEIGHT
    return prior
