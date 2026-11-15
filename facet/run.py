"""Data in, submission out:  python -m facet.run  ->  evaluation/results/facet.csv"""

import os
import subprocess
import sys

import numpy as np

from facet import decode, evidence, load, players
from facet.ball import fuse
from games import GAME, half_video_blocks, require
from starter.baseline import team_names, write_submission

CERTAIN = 10.0     # evidence strong enough that the decoder cannot disagree
OUT = os.path.join(load.ROOT, "evaluation", "results", "facet.csv")


def predict(game=GAME):
    tac = load.load("tactical", game)
    people, cut = players.drop_ghosts(tac["players"])
    ball = fuse.ball_track(game)
    home_name = team_names(require(["atd_meta"], game)[0])[0]
    team = (evidence.team_evidence(ball, people, tac["team"])
            + evidence.tag_prior(len(people), home_name, game))
    dead = evidence.dead_evidence(ball, people, game)
    dead[evidence.before_kickoff(len(dead), game)] = CERTAIN

    labels = np.full(len(team), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        path = decode.viterbi(team[first:last + 1], dead[first:last + 1])
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]
    print("broadcast offsets {}  ghost frames cut {}  ball seen {:.1%}  disputed {:.1%}".format(
        ball["offsets"], cut, ball["seen"].mean(), ball["disputed"].mean()))
    return labels


def describe(labels, game=GAME):
    """Label-free sanity numbers: a professional match has roughly 350 possessions of ~14 s."""
    spells = dead_frames = graded = 0
    for _, first, last in half_video_blocks(game):
        part = labels[first:last + 1]
        starts = np.r_[True, part[1:] != part[:-1]]
        spells += int((starts & (part != "D")).sum())
        dead_frames += int((part == "D").sum())
        graded += len(part)
    live = graded - dead_frames
    print("possessions {}  mean {:.1f} s  dead {:.1%}".format(
        spells, live / max(spells, 1) / load.FPS, dead_frames / graded))


if __name__ == "__main__":
    result = predict()
    describe(result)
    write_submission(OUT, result)
    print("wrote", os.path.relpath(OUT, load.ROOT))
    for script in ("validate_submission.py", "run_evaluation.py"):
        args = [OUT] if script.startswith("validate") else []
        subprocess.run([sys.executable, os.path.join(load.ROOT, "evaluation", script)] + args,
                       check=True)
