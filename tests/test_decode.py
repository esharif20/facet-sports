import numpy as np

from facet.decode import A, D, H, emissions, viterbi


def decode(team, dead):
    return viterbi(emissions(np.asarray(team, float), np.asarray(dead, float)), 5.0, 30.0)


def test_flicker_does_not_switch_team():
    team = np.full(200, 1.0)
    team[100:103] = -2.0                      # 3 frames of away evidence: a touch, not control
    assert (decode(team, np.full(200, -3.0)) == H).all()


def test_sustained_control_switches_team():
    team = np.r_[np.full(100, 1.0), np.full(100, -1.0)]
    path = decode(team, np.full(200, -3.0))
    assert (path[:90] == H).all() and (path[110:] == A).all()


def test_dead_stretch():
    dead = np.r_[np.full(80, -3.0), np.full(80, 3.0), np.full(80, -3.0)]
    path = decode(np.ones(240), dead)
    assert (path[100:140] == D).all() and path[0] == H and path[-1] == H
