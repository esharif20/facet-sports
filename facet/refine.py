"""A second look at ownership: the first models' answers over the surrounding seconds.

The ownership and dead-ball models judge one frame from hand-made measurements. This model reads
their answers up to 7 s either side, and how long each run of answers for the same team lasts,
and corrects the frame's ownership. It learns, for example, that a second of weak evidence for
one team between two long spells of the other is usually not a possession.

It is trained on answers the first models gave for matches they had not seen, so it learns from
the mistakes they make on a new match.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from facet import dead, owner, sample_games

WINDOWS = (7, 25, 75, 175)      # frames
RUNS = (7, 25)                  # frames of smoothing before the answers are cut into runs
LONGEST = 1500                  # frames: run lengths are capped here
FOLDS = 5
TRAIN_EVERY = 8
MAX_LLR = 6.0


def _runs(side, value):
    """For each frame, the run of equal `side` it sits in: (lengths, means). Lengths in seconds
    of this run, the part gone, the part left, the run before and the run after; means of
    `value` over this run, the one before and the one after."""
    change = np.r_[True, side[1:] != side[:-1]]
    run, starts = np.cumsum(change) - 1, np.where(change)[0]
    length = np.diff(np.r_[starts, len(side)]).astype(float)
    mean = np.add.reduceat(value, starts) / length
    gone = np.arange(len(side)) - starts[run]
    lengths = [length[run], gone, length[run] - gone, np.r_[0, length[:-1]][run],
               np.r_[length[1:], 0][run]]
    return [np.minimum(v, LONGEST) / dead.SECOND for v in lengths], \
        [mean[run], np.r_[0, mean[:-1]][run], np.r_[mean[1:], 0][run]]


def context(team_llr, dead_llr, seen):
    """(signed [T,S], plain [T,P]). Signed features flip when the teams are swapped."""
    team_llr, dead_llr = team_llr.astype(np.float64), dead_llr.astype(np.float64)
    seen = seen.astype(float)
    signed, plain = [team_llr], [dead_llr, dead._smooth(seen, 51), dead._smooth(seen, 251)]
    for frames in WINDOWS:
        team, stopped = dead._smooth(team_llr, frames), dead._smooth(dead_llr, frames)
        signed += [team, dead._shift(team, -frames), dead._shift(team, frames)]
        plain += [stopped, dead._shift(stopped, -frames), dead._shift(stopped, frames)]
    for frames in RUNS:
        team = dead._smooth(team_llr, frames)
        side = np.sign(team)
        lengths, means = _runs(side, team)
        plain += lengths
        signed += means + [side * lengths[0], side * lengths[3], side * lengths[4]]
    stopped = dead._smooth(dead_llr, 13)
    lengths, means = _runs(stopped > 0, stopped)
    plain += lengths + means + [(stopped > 0) * lengths[0]]
    return np.stack(signed, 1).astype(np.float32), np.stack(plain, 1).astype(np.float32)


def training_set(game_numbers=sample_games.TRAINING):
    inputs, answers = [], []
    for fold in range(FOLDS):
        held = game_numbers[fold::FOLDS]
        rest = [number for number in game_numbers if number not in held]
        owns, stops = owner.fit(rest), dead.fit(rest)
        for number in held:
            game = sample_games.training_copies(number)[0]
            ball = game["ball_xy"]
            seen = ~np.isnan(ball[:, 0])
            signed, plain = context(
                owner.team_evidence(owns, game["players"], game["team"], ball, seen),
                dead.dead_evidence(stops, game["players"], ball, sample_games.PITCH), seen)
            live = np.where(game["labels"] != "D")[0][::TRAIN_EVERY]
            home = game["labels"][live] == "H"
            inputs += [np.c_[signed[live], plain[live]], np.c_[-signed[live], plain[live]]]
            answers += [home, ~home]                         # and the same with teams swapped
    return np.concatenate(inputs), np.concatenate(answers)


def fit(game_numbers=sample_games.TRAINING):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=100,
                                           l2_regularization=1.0, random_state=0)
    return model.fit(*training_set(tuple(game_numbers)))


def team_evidence(model, team_llr, dead_llr, seen):
    """log P(home owns) - log P(away owns) per frame [T], made exactly symmetric."""
    signed, plain = context(team_llr, dead_llr, seen)
    p = (model.predict_proba(np.c_[signed, plain])[:, 1]
         + 1 - model.predict_proba(np.c_[-signed, plain])[:, 1]) / 2
    return np.clip(np.log(p + 1e-6) - np.log(1 - p + 1e-6), -MAX_LLR, MAX_LLR)
