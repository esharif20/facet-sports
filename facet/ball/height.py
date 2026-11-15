"""Ball height from two cameras.

Each feed places the ball where its camera's line of sight meets the ground. A ball in the air
therefore lands on a different ground point for each camera. With both camera positions known, the
two lines of sight are intersected to get the real 3D ball.
"""

import numpy as np

from games import GAME, GAMES

IMAGE_CENTRE = (960.0, 540.0)
SAME_CAMERA = np.array([8.0, 15.0, 8.0])   # m: a broadcast frame this close to the main camera
MAX_MISS = 1.5                             # m: the lines must pass this close to each other
MIN_HEIGHT, MAX_HEIGHT = -1.5, 40.0        # m: outside this the answer is rejected


def camera_centres(matrices, game=GAME):
    """Camera position [N,3] in metres from each frame's pitch->pixel homography.

    Assumes square pixels and the optical axis through the image centre, which leaves the focal
    length as the only unknown; it follows from the two pitch axes being perpendicular.
    """
    length, width = GAMES[game]["pitch"]
    a = matrices @ np.diag([1 / length, 1 / width, 1.0])          # metres -> pixels
    a[:, 0, :] -= IMAGE_CENTRE[0] * a[:, 2, :]
    a[:, 1, :] -= IMAGE_CENTRE[1] * a[:, 2, :]
    a1, a2, a3 = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        by_angle = -(a1[:, 0] * a2[:, 0] + a1[:, 1] * a2[:, 1]) / (a1[:, 2] * a2[:, 2])
        by_length = ((a2[:, 0] ** 2 + a2[:, 1] ** 2 - a1[:, 0] ** 2 - a1[:, 1] ** 2)
                     / (a1[:, 2] ** 2 - a2[:, 2] ** 2))
        # use whichever constraint is better conditioned for this view
        focal2 = np.where(np.abs(a1[:, 2] ** 2 - a2[:, 2] ** 2) > np.abs(a1[:, 2] * a2[:, 2]),
                          by_length, by_angle)
        focal = np.sqrt(np.where(focal2 > 0, focal2, np.nan))

        def unscale(v):
            return np.stack([v[:, 0] / focal, v[:, 1] / focal, v[:, 2]], axis=1)

        r1, r2, t = unscale(a1), unscale(a2), unscale(a3)
        scale = 1 / np.linalg.norm(r1, axis=1, keepdims=True)
        r1, t = r1 * scale, t * scale
        r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)
    r3 = np.cross(r1, r2)
    centre = -np.stack([(r1 * t).sum(1), (r2 * t).sum(1), (r3 * t).sum(1)], axis=1)
    centre[:, 2] = np.abs(centre[:, 2])                           # the camera is above the pitch
    return centre


def triangulate(ground_a, camera_a, ground_b, camera_b):
    """3D point nearest both lines of sight [N,3], and how far apart the lines pass [N]."""
    g1 = np.c_[ground_a, np.zeros(len(ground_a))]
    g2 = np.c_[ground_b, np.zeros(len(ground_b))]
    d1, d2, w = camera_a - g1, camera_b - g2, g1 - g2
    a, b, c = (d1 * d1).sum(1), (d1 * d2).sum(1), (d2 * d2).sum(1)
    d, e = (d1 * w).sum(1), (d2 * w).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s, u = (b * e - c * d) / (a * c - b * b), (a * e - b * d) / (a * c - b * b)
    p1, p2 = g1 + s[:, None] * d1, g2 + u[:, None] * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2, axis=1)


def ball_3d(tac_ball, bro_ball, tac_centres, bro_centres, in_play):
    """Ball position [T,3] with real height, NaN where the two cameras cannot give one.

    tac_ball, bro_ball: ground positions [T,2] on tactical frames. *_centres: per-frame camera
    positions [T,3]. in_play: frames used to locate the fixed tactical camera.
    """
    tac_camera = np.nanmedian(tac_centres[in_play], axis=0)
    bro_camera = np.nanmedian(bro_centres, axis=0)                # the main broadcast camera
    main = (np.abs(bro_centres - bro_camera) < SAME_CAMERA).all(1)
    ok = main & ~np.isnan(tac_ball[:, 0]) & ~np.isnan(bro_ball[:, 0])

    out = np.full((len(tac_ball), 3), np.nan)
    point, miss = triangulate(tac_ball[ok], tac_camera, bro_ball[ok], bro_camera)
    good = (miss < MAX_MISS) & (point[:, 2] > MIN_HEIGHT) & (point[:, 2] < MAX_HEIGHT)
    out[np.where(ok)[0][good]] = point[good]
    return out, tac_camera, bro_camera
