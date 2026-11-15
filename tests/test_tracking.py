import numpy as np

from facet import players, sync
def test_sync_finds_the_offset_of_each_half():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 0.5, (199319, 2)), axis=0).astype(np.float32)
    walk[rng.random(len(walk)) < 0.3] = np.nan
    broadcast = np.r_[walk[12000:60000], walk[120000:170000]]
    mapping, offsets = sync.broadcast_to_tactical(walk, broadcast)
    assert offsets == [12000, 72000]
    assert mapping[100] == 12100 and mapping[48000 + 100] == 120100


def test_ghost_is_removed_and_duel_is_kept():
    people = np.full((400, 3, 3), np.nan, dtype=np.float32)
    people[:, 0, :2] = [10.0, 10.0]                     # a real player, seen throughout
    people[100:200, 1, :2] = [10.3, 10.0]               # a ghost: its whole life is next to them
    people[:, 2, :2] = [40.0, 40.0]                     # an opponent, also seen throughout...
    people[250:330, 2, :2] = [10.5, 10.0]               # ...who comes in for a duel
    cleaned, cut = players.drop_ghosts(people)
    assert np.isnan(cleaned[100:200, 1, 0]).all() and cut == 100
    assert not np.isnan(cleaned[:, [0, 2], 0]).any()
