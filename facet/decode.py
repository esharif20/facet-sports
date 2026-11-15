"""Viterbi over three states (H, A, D) with a fixed cost for changing state.

The cost is what turns flickering per-frame evidence into whole possessions: a change of team
only pays off when the new team's evidence, summed over frames, beats SWITCH_TEAM.
"""

import numpy as np

H, A, D = 0, 1, 2
# SWITCH_TEAM is in frames of clear control: the other team must hold the ball for about 10 frames
# (0.4 s) more than it costs, the control duration used in the tracking literature. SWITCH_DEAD
# means a stoppage shorter than about a second is not believed. Neither was fitted to ground truth.
SWITCH_TEAM = 10.0
SWITCH_DEAD = 25.0


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
