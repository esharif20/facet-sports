"""Map broadcast frames onto tactical frames.

The broadcast recording has its own timeline. Both feeds track the same ball on the same pitch, so
a stretch of the broadcast ball track is slid along the tactical one until they agree. The offset
is constant within a half (the broadcast only cuts at halftime), so each confident stretch votes
and each half takes the median.
"""

import numpy as np

from games import GAME, half_video_blocks

WINDOW = 1500          # broadcast frames per stretch (60 s)
MIN_BALL = 500         # ball frames a stretch needs before it may vote
MIN_OVERLAP = 300      # frames with the ball in both feeds at a candidate offset
CONFIDENT = 2.0        # runner-up error / best error: below this the stretch does not vote
MIN_VOTES = 5


def _correlate(big_fft, small, n):
    """sum_i big[i + k] * small[i] for every shift k."""
    return np.fft.irfft(big_fft * np.conj(np.fft.rfft(small, n)), n)


def window_offsets(tac_ball, bro_ball):
    """[(broadcast start, offset, confidence)] for every stretch that can be matched."""
    n = 1 << int(np.ceil(np.log2(len(tac_ball) + WINDOW)))
    tac_seen = ~np.isnan(tac_ball[:, 0])
    tac = np.nan_to_num(tac_ball[:, :2]).astype(np.float64)
    f_seen = np.fft.rfft(tac_seen.astype(float), n)
    f_pos = [np.fft.rfft(tac[:, d], n) for d in (0, 1)]
    f_sq = [np.fft.rfft(tac[:, d] ** 2, n) for d in (0, 1)]

    votes = []
    for start in range(0, len(bro_ball) - WINDOW, WINDOW):
        piece = bro_ball[start:start + WINDOW, :2].astype(np.float64)
        seen = (~np.isnan(piece[:, 0])).astype(float)
        if seen.sum() < MIN_BALL:
            continue
        piece = np.nan_to_num(piece)
        # squared distance between the two tracks, summed over frames where both see the ball
        error = sum(_correlate(f_sq[d], seen, n) - 2 * _correlate(f_pos[d], piece[:, d], n)
                    + _correlate(f_seen, piece[:, d] ** 2, n) for d in (0, 1))
        overlap = _correlate(f_seen, seen, n)
        mean = np.where(overlap > MIN_OVERLAP, error / np.maximum(overlap, 1), np.inf)
        mean = mean[:len(tac_ball) - WINDOW]
        best = int(np.argmin(mean))
        rest = mean.copy()
        rest[max(0, best - 100):best + 100] = np.inf
        votes.append((start, best - start, float(np.sqrt(rest.min() / max(mean[best], 1e-9)))))
    return votes


def _refine(tac_ball, bro_ball, offset, reach=5):
    """The offset within +-reach frames where the two ball tracks sit closest together."""
    frames = np.arange(len(bro_ball))

    def spread(candidate):
        ok = (frames + candidate >= 0) & (frames + candidate < len(tac_ball))
        gap = tac_ball[frames[ok] + candidate, :2] - bro_ball[frames[ok], :2]
        return np.nanmedian(np.linalg.norm(gap, axis=1))

    return min(range(offset - reach, offset + reach + 1), key=spread)


def broadcast_to_tactical(tac_ball, bro_ball, game=GAME):
    """Tactical frame of every broadcast frame, -1 where there is none. Also the offsets found."""
    votes = [(s, off) for s, off, conf in window_offsets(tac_ball, bro_ball) if conf >= CONFIDENT]
    offsets = np.array(sorted(off for _, off in votes))
    groups = np.split(offsets, np.where(np.diff(offsets) > 50)[0] + 1)
    found = [_refine(tac_ball, bro_ball, int(np.median(g))) for g in groups if len(g) >= MIN_VOTES]

    # every broadcast frame follows the confident stretch nearest to it in time
    votes = [(s, off) for s, off in votes if min(abs(off - f) for f in found) <= 50]
    starts = np.array([s for s, _ in votes])
    group = np.array([int(np.argmin([abs(off - f) for f in found])) for _, off in votes])
    frames = np.arange(len(bro_ball))
    nearest = group[np.abs(frames[:, None] - (starts + WINDOW // 2)[None, :]).argmin(1)]
    mapped = frames + np.array(found)[nearest]
    mapping = np.full(len(bro_ball), -1)
    for index, offset in enumerate(found):
        centres = starts[group == index] + WINDOW // 2 + offset
        # the half this group plays in: where most of its stretches land
        _, first, last = max(half_video_blocks(game),
                             key=lambda b: ((centres >= b[1]) & (centres <= b[2])).sum())
        inside = (nearest == index) & (mapped >= first) & (mapped <= last)
        mapping[inside] = mapped[inside]
    return mapping, found


def onto_tactical(values, mapping, n_tactical):
    """Broadcast rows moved to tactical frames; NaN where the broadcast has nothing."""
    out = np.full((n_tactical,) + values.shape[1:], np.nan, dtype=np.float64)
    out[mapping[mapping >= 0]] = values[mapping >= 0]
    return out
