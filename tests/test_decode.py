import numpy as np

from facet.decode import viterbi, H, A, D


def test_flicker_does_not_switch_team():
    team = np.full(200, 1.0)
    team[100:104] = -2.0                      # 4 frames of weak away evidence: a touch, not control
    assert (viterbi(team, np.full(200, -1.0)) == H).all()


def test_sustained_control_switches_team():
    team = np.r_[np.full(100, 1.0), np.full(100, -1.0)]
    path = viterbi(team, np.full(200, -1.0))
    assert (path[:90] == H).all() and (path[110:] == A).all()


def test_dead_stretch():
    dead = np.r_[np.full(80, -1.0), np.full(80, 2.0), np.full(80, -1.0)]
    path = viterbi(np.ones(240), dead)
    assert (path[100:140] == D).all() and path[0] == H and path[-1] == H
