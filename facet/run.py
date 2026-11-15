"""Data in, submission out:  python -m facet.run  ->  evaluation/results/facet.csv"""

import os
import subprocess
import sys

import numpy as np

from facet import dead, decode, load, owner, players, restarts
from facet.ball import fuse
from games import GAME, GAMES, half_video_blocks
from starter.baseline import write_submission

OUT = os.path.join(load.ROOT, "evaluation", "results", "facet.csv")


def predict(game=GAME):
    tac = load.load("tactical", game)
    people, cut = players.drop_ghosts(tac["players"])
    ball = fuse.ball_track(game)
    usable = ball["seen"] & ~ball["disputed"]

    team = owner.team_evidence(owner.fit(), people, tac["team"], ball["xy"], usable)
    stopped = dead.dead_evidence(dead.fit(), people, ball["xy"], GAMES[game]["pitch"])
    tagged = restarts.from_smart_tagging(game)
    tagged = restarts.with_throw_in_sides(tagged, people, tac["team"], GAMES[game]["pitch"])
    add_team, add_dead = restarts.evidence(len(team), tagged)
    team, stopped = team + add_team, stopped + add_dead
    stopped[dead.before_kickoff(len(stopped), game)] = 1000.0     # warm-up is certainly dead
    goals = restarts.goals_from_smart_tagging(game)
    stopped[restarts.goal_stoppages(len(stopped), goals)] = 1000.0   # so is a goal celebration
    emit = decode.emissions(team, stopped)

    labels = np.full(len(team), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = np.array(["H", "A", "D"])[decode.viterbi(emit[first:last + 1])]
    print("broadcast offsets {}  ghost frames cut {}  ball seen {:.1%}  disputed {:.1%}".format(
        ball["offsets"], cut, ball["seen"].mean(), ball["disputed"].mean()))
    throw_ins = [side for _, side, kind in tagged if restarts.is_throw_in(kind)]
    print("smart tagging restarts {}  with a team {}  throw-ins {} (side from tracking {})  "
          "goals {} (paired with a GOALS tag {})".format(
              len(tagged), sum(side is not None for _, side, _ in tagged), len(throw_ins),
              sum(side is not None for side in throw_ins), len(goals),
              sum(goal is not None for goal, _ in goals)))
    return labels


def describe(labels, game=GAME):
    """Label-free sanity numbers: possessions, their mean length, and the dead share."""
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
