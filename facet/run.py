"""Data in, submission out:  python -m facet.run  ->  evaluation/results/facet.csv"""

import os
import subprocess
import sys

import numpy as np

from facet import dead, decode, features, load, possession
from games import half_video_blocks
from starter.baseline import write_submission

ROOT = load.ROOT
OUT = os.path.join(ROOT, "evaluation", "results", "facet.csv")


def predict():
    f = features.build(load.load())
    team, dead_llr = possession.team_evidence(f), dead.dead_evidence(f)
    labels = np.full(len(team), "X", dtype="<U1")
    for _, first, last in half_video_blocks("BAR-ATH"):
        path = decode.viterbi(team[first:last + 1], dead_llr[first:last + 1])
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]
    return labels


if __name__ == "__main__":
    write_submission(OUT, predict())
    print("wrote", os.path.relpath(OUT, ROOT))
    for script in ("validate_submission.py", "run_evaluation.py"):
        args = [OUT] if script.startswith("validate") else []
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", script)] + args, check=True)
