"""Viterbi over three states (H, A, D) with a fixed cost for changing state.

The cost is what turns flickering per-frame evidence into whole possessions: a change of team
only pays off when the new team's evidence, summed over frames, beats SWITCH_TEAM.
"""

import numpy as np

H, A, D = 0, 1, 2
# Chosen from football priors checked on the unlabelled full match (no ground truth): they give
# ~400 possessions of ~11 s and ~85 stoppages, in the range a professional match has.
SWITCH_TEAM = 20.0
SWITCH_DEAD = 40.0


def viterbi(team_llr, dead_llr, switch_team=SWITCH_TEAM, switch_dead=SWITCH_DEAD):
    """Best state path [T] for one half. team_llr favours H when positive, dead_llr favours D."""
    n = len(team_llr)
    emit = np.stack([team_llr / 2, -team_llr / 2, dead_llr], axis=1)
    cost = np.array([[0, switch_team, switch_dead],
                     [switch_team, 0, switch_dead],
                     [switch_dead, switch_dead, 0]], dtype=float)
    back = np.empty((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost          # [from, to]
        back[t] = cand.argmax(0)
        score = cand.max(0) + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
