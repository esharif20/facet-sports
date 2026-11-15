"""Score the pipeline on Metrica's open sample games, with the official metric.

This is where settings are chosen. The competition match's ground truth is never used for that.
Each game is first made as rough as our tracking, then scored with models trained on the other
game.

    python -m facet.bench          the current settings
    python -m facet.bench sweep    a grid around them
    python -m facet.bench restarts with and without Smart Tagging restarts (facet.restarts)
"""

import itertools
import os
import sys

import numpy as np

from facet import dead, decode, owner, restarts, sample_games
from facet.load import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def prepare(number):
    """One game's evidence, from models that never saw it."""
    others = [n for n in sample_games.GAMES if n != number]
    game = sample_games.like_ours(sample_games.load(number), seed=100 + number)
    xy = game["ball_xy"]
    game.update(
        dead=dead.dead_evidence(dead.fit(others), game["players"], xy, sample_games.PITCH),
        owner=owner.team_evidence(owner.fit(others), game["players"], game["team"], xy,
                                  ~np.isnan(xy[:, 0])),
        tagged=restarts.like_smart_tagging(game["restarts"], game["labels"], seed=200 + number))
    return game


def score(game, switch_team=decode.SWITCH_TEAM, switch_dead=decode.SWITCH_DEAD,
          team_scale=decode.TEAM_SCALE, dead_scale=decode.DEAD_SCALE, use_restarts=False,
          restart_dead=restarts.DEAD_WEIGHT, restart_team=restarts.TEAM_WEIGHT):
    team, stopped = game["owner"], game["dead"]
    if use_restarts:
        add_team, add_dead = restarts.evidence(len(team), game["tagged"], restart_dead,
                                               restart_team)
        team, stopped = team + add_team, stopped + add_dead
    emit = decode.emissions(team, stopped, team_scale, dead_scale)
    truth, guess = [], []
    for period in (1, 2):
        rows = game["period"] == period
        path = decode.viterbi(emit[rows], switch_team, switch_dead)
        truth += list(game["labels"][rows]) + ["X"]        # X keeps the halves apart
        guess += list(np.array(["H", "A", "D"])[path]) + ["X"]
    result = pm.evaluate(pm.label_codes(np.array(truth)), pm.label_codes(np.array(guess)))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], \
        result["n_gt_spells"]


def report(games, **settings):
    rows = [score(game, **settings) for game in games]
    print("{:<78s} spellF1 {:.3f}  macroF1 {:.3f}  possessions {} / {}".format(
        str(settings), np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows]),
        sum(r[2] for r in rows), sum(r[3] for r in rows)), flush=True)


if __name__ == "__main__":
    prepared = [prepare(number) for number in sample_games.GAMES]
    report(prepared)
    report(prepared, use_restarts=True)          # what facet.run does: with Smart Tagging restarts
    if sys.argv[1:] == ["sweep"]:
        for values in itertools.product((3, 5, 8, 12), (30, 60, 100), (0.5, 1.0), (0.5, 1.0)):
            report(prepared, **dict(zip(("switch_team", "switch_dead", "team_scale",
                                         "dead_scale"), values)))
    if sys.argv[1:] == ["restarts"]:
        for dead_weight, team_weight in itertools.product((0.5, 1.0, 2.0, 4.0),
                                                          (0.0, 0.5, 1.0, 2.0)):
            report(prepared, use_restarts=True, restart_dead=dead_weight, restart_team=team_weight)
