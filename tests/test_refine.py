import numpy as np

from facet import refine


def test_swapping_the_teams_flips_signed_features_only():
    rng = np.random.default_rng(0)
    team = np.cumsum(rng.normal(0, 0.3, 2000)).clip(-4, 4)
    stopped = np.cumsum(rng.normal(0, 0.3, 2000)).clip(-4, 4)
    seen = rng.random(2000) < 0.8
    signed, plain = refine.context(team, stopped, seen)
    flipped, same = refine.context(-team, stopped, seen)
    assert np.allclose(flipped, -signed) and np.allclose(same, plain)


def test_runs_measure_the_run_a_frame_sits_in():
    side = np.array([1, 1, 1, -1, -1, 1])
    lengths, means = refine._runs(side, side.astype(float))
    seconds = np.array([3, 3, 3, 2, 2, 1]) / 25
    assert np.allclose(lengths[0], seconds)                  # this run
    assert np.allclose(lengths[3][3:5], 3 / 25)              # the run before
    assert np.allclose(means[0], side)
