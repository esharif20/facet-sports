import numpy as np

from facet import players, sync
from facet.ball import height

CAMERA = np.array([50.0, 110.0, 25.0])


def homography_from(camera, focal=2500.0):
    """Pitch (0-1) -> pixel matrix of a camera at `camera` looking at the pitch centre."""
    forward = np.array([52.65, 34.0, 0.0]) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    k = np.array([[focal, 0, 960.0], [0, focal, 540.0], [0, 0, 1.0]])
    metres = k @ np.c_[rotation[:, 0], rotation[:, 1], -rotation @ camera]
    return metres @ np.diag([105.3, 68.0, 1.0])


def test_camera_position_recovered_from_homography():
    found = height.camera_centres(homography_from(CAMERA)[None])[0]
    assert np.allclose(found, CAMERA, atol=1e-6)


def test_two_sight_lines_give_the_height():
    ball = np.array([[30.0, 20.0, 6.0]])
    other = np.array([60.0, 112.0, 18.0])

    def shadow(camera):                       # where the camera's line of sight meets the ground
        return (camera + (ball - camera) * camera[2] / (camera[2] - ball[:, 2:]))[:, :2]

    point, miss = height.triangulate(shadow(CAMERA), CAMERA, shadow(other), other)
    assert np.allclose(point, ball, atol=1e-6) and miss[0] < 1e-6


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
