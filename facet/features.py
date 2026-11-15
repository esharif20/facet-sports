"""Per-frame features from the cached tracking. Every array is [T], NaN where undefined."""

import warnings

import numpy as np

from games import GAME, GAMES

FPS = 25


def build(data, game=GAME):
    players, ball, team = data["players"], data["ball"], data["team"]
    length, width = GAMES[game]["pitch"]

    dist = np.sqrt(((players[:, :, :2] - ball[:, None, :2]) ** 2).sum(-1))
    with warnings.catch_warnings():  # frames with no tracked player of a team are NaN, on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        near_home = np.nanmin(np.where(team == 0, dist, np.nan), 1)
        near_away = np.nanmin(np.where(team == 1, dist, np.nan), 1)
        player_speed = np.nanmean(players[:, :, 2], 1)

    ball_speed = np.full(len(ball), np.nan)
    ball_speed[1:] = np.sqrt(((ball[1:, :2] - ball[:-1, :2]) ** 2).sum(-1)) * FPS

    bx, by = ball[:, 0], ball[:, 1]
    return {
        "ball_seen": ~np.isnan(bx),
        "near_home": near_home,                     # m, nearest home player to the ball
        "near_away": near_away,                     # m, nearest away player to the ball
        "ball_speed": ball_speed,                   # m/s, from consecutive positions
        "ball_out": (bx < -1) | (bx > length + 1) | (by < -1) | (by > width + 1),
        "player_speed": player_speed,               # m/s, mean over tracked players
    }
