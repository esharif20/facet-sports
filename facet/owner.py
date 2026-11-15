"""Which team owns the ball? A classifier trained on other matches.

Ownership is decided by contacts: stretches where one team has a player within reach of the
ball. Whether a contact takes the ball (a controlled touch) or not (a deflection, a ball flying
past) depends on how long it lasts and what comes before and after it. Those rules are learned
from Metrica's open sample games (see facet.sample_games), where events name the owner.

Every feature is either symmetric or flips sign when the teams are swapped, and the model is
trained on both versions, so it cannot favour a side.
"""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from facet import dead, sample_games
from facet.load import FPS

REACH = 1.5             # m: a player this close is in contact with the ball
FAR = 8.0               # m: distances are capped here
MERGE = 3               # frames: holes this short do not split a contact
TOUCH_SPAN = 5          # frames either side over which the ball's velocity is averaged
TOUCH_JOLT = 4.0        # m/s: a change of velocity this large is a touch
TRAIN_EVERY = 3
MAX_LLR = 4.0


def _velocity(xy, half_window=2):
    out = np.full(xy.shape, np.nan)
    out[half_window:-half_window] = (xy[2 * half_window:] - xy[:-2 * half_window]) \
        * (FPS / (2 * half_window))
    return out


def _nearest(players, team, ball_xy, side):
    """Distance from the ball to the nearest player of one side, and that player's speed
    relative to the ball [T], NaN where unknown."""
    xy = players[:, team == side, :2].astype(np.float64)
    offset = xy - ball_xy[:, None, :]
    dist = np.hypot(offset[:, :, 0], offset[:, :, 1])
    filled = np.where(np.isnan(dist), np.inf, dist)
    pick = filled.argmin(1)
    rows = np.arange(len(xy))
    rel = _velocity(xy)[rows, pick] - _velocity(ball_xy)
    best = filled[rows, pick]
    return np.where(np.isinf(best), np.nan, best), np.hypot(rel[:, 0], rel[:, 1])


def _flow(players, team):
    """Which way play is flowing: the team with the ball pushes everyone towards the other goal.

    Each half is split in two blocks by where the teams sit on average, which gives the direction
    home attacks in. Returns the players' collective drift along that direction, over 1 s and 4 s.
    """
    x = players[:, :, 0].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_x, away_x = np.nanmean(x[:, team == 0], axis=1), np.nanmean(x[:, team == 1], axis=1)
        centre = np.nanmean(x, axis=1)
    # home attacks towards larger x while it sits at smaller x; smoothing over ten minutes
    # follows the change of ends at halftime
    side = np.sign(dead._smooth(np.nan_to_num(away_x - home_x), 15000 + 1))
    out = []
    for frames in (25, 100):
        drift = np.full(len(x), np.nan)
        drift[frames:] = (centre[frames:] - centre[:-frames]) * FPS / frames
        drift = np.nan_to_num(drift)
        out += [side * drift, side * dead._shift(drift, frames)]      # just gone, and coming
    return out


def _shape(players, team, ball_x):
    """How the two teams sit, which says who has the ball when it cannot be seen.

    The team with the ball spreads out and the other closes up, and play sits in the half of the
    team without it. Every value is away minus home, or measured along home's attack direction.
    """
    x, y, speed = (players[:, :, i].astype(np.float64) for i in range(3))
    home, away = team == 0, team == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(x, axis=1)
        side = np.sign(dead._smooth(np.nan_to_num(                       # as in _flow
            np.nanmean(x[:, away], axis=1) - np.nanmean(x[:, home], axis=1)), 15000 + 1))
        middle = np.nanmedian(centre)
        gaps = [np.nanstd(x[:, away], axis=1) - np.nanstd(x[:, home], axis=1),
                np.nanstd(y[:, away], axis=1) - np.nanstd(y[:, home], axis=1),
                np.nanmean(speed[:, away], axis=1) - np.nanmean(speed[:, home], axis=1)]
    territory = np.nan_to_num(side * (centre - middle))
    ball = side * (ball_x - middle)
    last = np.maximum.accumulate(np.where(~np.isnan(ball), np.arange(len(ball)), 0))
    out = [territory, dead._smooth(territory, 5 * FPS), np.nan_to_num(ball[last])]
    for gap in gaps:
        gap = np.nan_to_num(gap)
        out += [dead._smooth(gap, 2 * FPS + 1), dead._smooth(gap, 10 * FPS + 1)]
    return out


def _touches(velocity, near_home, near_away):
    """Moments the ball is played, and who was nearest when it was.

    The tracked ball strays from the real one, but it strays slowly, so its velocity is still
    accurate. A sudden change of velocity is a touch, timed to a few frames. Returns signed and
    plain features about the two touches before and the two after each frame.
    """
    n = len(velocity)
    before, after = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    window = np.stack([velocity[lag:n - 2 * TOUCH_SPAN + lag] for lag in range(TOUCH_SPAN)])
    mean = window.mean(0)                                    # mean[i]: frames i .. i+SPAN-1
    before[TOUCH_SPAN:TOUCH_SPAN + len(mean)] = mean         # the SPAN frames up to t-1
    after[:len(mean) - 1] = mean[1:]                         # the SPAN frames from t+1
    jolt = np.nan_to_num(np.hypot(*(after - before).T))
    peak = (jolt > TOUCH_JOLT) & (jolt == np.max(
        np.stack([np.roll(jolt, shift) for shift in range(-TOUCH_SPAN, TOUCH_SPAN + 1)]), axis=0))
    at = np.where(peak)[0]
    side = np.sign(near_away[at] - near_home[at])
    margin = np.clip(near_away[at] - near_home[at], -5, 5)
    frames = np.arange(n)
    k = np.searchsorted(at, frames, side="right") - 1

    def of(index, values, default):
        ok = (index >= 0) & (index < len(at))
        return np.where(ok, values[np.clip(index, 0, max(len(at) - 1, 0))], default) if len(at) \
            else np.full(n, default)

    signed, plain = [], []
    for step in (-1, 0, 1, 2):
        signed += [of(k + step, side, 0.0), of(k + step, margin, 0.0)]
        plain += [of(k + step, jolt[at], 0.0),
                  np.clip(np.abs(of(k + step, at.astype(float), 10.0 ** 9) - frames), 0, 500),
                  of(k + step, np.minimum(near_home, near_away)[at], FAR)]
    return signed, plain


def contacts(closeness, reach_ok):
    """Stretches where one side is in contact: arrays (start, end, side) with side +1 home."""
    side = np.where(reach_ok, np.sign(closeness), 0).astype(int)
    last = 0
    for t in range(len(side)):                           # close short holes inside one contact
        if side[t] != 0:
            if 0 < t - last <= MERGE and side[last] == side[t]:
                side[last:t] = side[t]
            last = t
    change = np.where(np.diff(np.r_[0, side, 0]) != 0)[0]
    starts, ends = change[:-1], change[1:]
    keep = side[starts] != 0
    return starts[keep], ends[keep], side[starts[keep]]


def features(players, team, ball_xy, usable):
    """(signed [T,S], plain [T,P]). Signed features flip when the teams are swapped."""
    ball = np.where(usable[:, None], ball_xy, np.nan).astype(np.float64)
    home, home_rel = _nearest(players, team, ball, 0)
    away, away_rel = _nearest(players, team, ball, 1)
    near_home = np.clip(np.nan_to_num(home, nan=FAR), 0, FAR)
    near_away = np.clip(np.nan_to_num(away, nan=FAR), 0, FAR)
    closeness = near_away - near_home                     # positive: home is nearer
    velocity = _velocity(ball)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    change = np.full(len(ball), np.nan)                   # how sharply the ball's velocity turns
    change[2:-2] = np.hypot(*(velocity[4:] - velocity[:-4]).T)

    starts, ends, sides = contacts(closeness, np.minimum(near_home, near_away) < REACH)
    n, frames = len(starts), np.arange(len(ball))
    k = np.searchsorted(starts, frames, side="right") - 1    # the contact in progress or just gone

    def of(index, values, default=0.0):
        ok = (index >= 0) & (index < n)
        return np.where(ok, values[np.clip(index, 0, max(n - 1, 0))], default) if n else \
            np.full(len(frames), default)

    length = np.minimum(ends - starts, 250).astype(float)
    rel_change = np.zeros(n)                                 # ball velocity change across it
    for i in range(n):
        before, after = velocity[max(starts[i] - 3, 0)], velocity[min(ends[i] + 2, len(ball) - 1)]
        rel_change[i] = np.nan_to_num(np.hypot(*(after - before)), nan=0.0)
    since = np.minimum(np.maximum(frames - of(k, ends, -500), 0), 500)
    until = np.minimum(np.maximum(of(k + 1, starts, 10 ** 9) - frames, 0), 500)

    signed = [closeness, dead._smooth(closeness, 5), dead._smooth(closeness, 13),
              dead._smooth(closeness, 25)]
    for lag in (12, 25, 50):
        wide = dead._smooth(closeness, 2 * lag + 1)
        signed += [dead._shift(wide, -lag), dead._shift(wide, lag)]
    signed += [np.nan_to_num(away_rel, nan=20.0).clip(0, 20) - np.nan_to_num(home_rel, nan=20.0)
               .clip(0, 20)]
    extra_signed, extra_plain = _touches(velocity, near_home, near_away)
    signed += extra_signed + _flow(players, team)
    plain = [np.minimum(near_home, near_away), np.nan_to_num(speed, nan=-1.0),
             np.nan_to_num(change, nan=-1.0), usable.astype(float), since, until]
    plain += extra_plain
    for step in (-2, -1, 0, 1, 2):                           # two contacts back to two ahead
        side = of(k + step, sides.astype(float))
        signed += [side, side * of(k + step, length) / 250.0]
        plain += [of(k + step, length), of(k + step, rel_change)]
    signed += _shape(players, team, ball[:, 0])
    return np.stack(signed, 1).astype(np.float32), np.stack(plain, 1).astype(np.float32)


def training_set(game_numbers=sample_games.GAMES):
    inputs, answers = [], []
    for number in game_numbers:
        game = sample_games.like_ours(sample_games.load(number), seed=number)
        ball = game["ball_xy"]
        signed, plain = features(game["players"], game["team"], ball, ~np.isnan(ball[:, 0]))
        live = np.where(game["labels"] != "D")[0][::TRAIN_EVERY]
        home = game["labels"][live] == "H"
        inputs += [np.c_[signed[live], plain[live]], np.c_[-signed[live], plain[live]]]
        answers += [home, ~home]                             # and the same with teams swapped
    return np.concatenate(inputs), np.concatenate(answers)


def fit(game_numbers=sample_games.GAMES):
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=200,
                                           l2_regularization=1.0, random_state=0)
    return model.fit(*training_set(game_numbers))


def team_evidence(model, players, team, ball_xy, usable):
    """log P(home owns) - log P(away owns) per frame [T], made exactly symmetric."""
    signed, plain = features(players, team, ball_xy, usable)
    p = (model.predict_proba(np.c_[signed, plain])[:, 1]
         + 1 - model.predict_proba(np.c_[-signed, plain])[:, 1]) / 2
    return np.clip(np.log(p + 1e-6) - np.log(1 - p + 1e-6), -MAX_LLR, MAX_LLR)
