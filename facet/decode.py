"""Viterbi over three states (H, A, D) with a fixed cost for changing state.

The cost is what turns flickering per-frame evidence into whole possessions: a change of state
only pays off when the evidence for it, summed over frames, beats the cost.
"""

import numpy as np

H, A, D = 0, 1, 2
# Chosen on training matches the models had not seen, never on the competition ground truth.
SWITCH_TEAM = 3.0
SWITCH_DEAD = 20.0
TEAM_SCALE = 0.5       # neighbouring frames repeat each other, so evidence may need shrinking
DEAD_SCALE = 0.5


def emissions(team_llr, dead_llr, team_scale=TEAM_SCALE, dead_scale=DEAD_SCALE):
    """Log-probability of each frame under H, A and D [T,3].

    team_llr: log P(home owns) - log P(away owns). dead_llr: log P(dead) - log P(alive).
    """
    team, dead = team_scale * team_llr, dead_scale * dead_llr
    alive = -np.logaddexp(0, dead)
    return np.stack([alive - np.logaddexp(0, -team), alive - np.logaddexp(0, team),
                     -np.logaddexp(0, -dead)], axis=1)


def viterbi(emit, switch_team=SWITCH_TEAM, switch_dead=SWITCH_DEAD):
    """Best state path [T] for one half, given emissions [T,3]."""
    cost = np.array([[0, switch_team, switch_dead],
                     [switch_team, 0, switch_dead],
                     [switch_dead, switch_dead, 0]], dtype=float)
    back = np.empty((len(emit), 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, len(emit)):
        cand = score[:, None] - cost          # [from, to]
        back[t] = cand.argmax(0)
        score = cand.max(0) + emit[t]
    path = np.empty(len(emit), dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(len(emit) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
