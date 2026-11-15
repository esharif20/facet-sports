"""Is the ball dead? A classifier trained on other matches.

The public ground truth shows that a dead ball is often still visible and tracked, so "ball
missing" alone is a poor test. What marks a stoppage is how everyone moves: players walk, nobody
sprints, the ball sits still or is carried. Those patterns are learned from Metrica's open sample
games (see facet.sample_games), where the events say when play stopped and restarted.

The sample games are first made as rough as our tracking (sample_games.like_ours), so the model
does not learn to rely on a ball that is cleaner than ours.
"""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from facet import sample_games
from facet.load import FPS
from games import GAME, half_video_blocks
from starter import load_smart_tagging

SECOND = FPS
TRAIN_EVERY = 5         # frames: neighbours are near-duplicates
MAX_LLR = 4.0


def _smooth(values, frames):
    """Mean over a centred window, ignoring NaN."""
    ok = ~np.isnan(values)
    kernel = np.ones(frames)
    total = np.convolve(np.where(ok, values, 0.0), kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    return total / np.maximum(count, 1)


def _shift(values, frames):
    """values[t + frames], edge-padded."""
    return values[np.clip(np.arange(len(values)) + frames, 0, len(values) - 1)]


def features(players, ball_xy, pitch):
    """[T,F] describing how players and ball move around each frame."""
    length, width = pitch
    speed = sample_games._speed(players[:, :, :2].astype(np.float64))
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):   # frames with nobody tracked
        warnings.simplefilter("ignore", RuntimeWarning)
        tracked = np.maximum((~np.isnan(speed)).sum(1), 1)     # NaN > 2 is False, so count by hand
        motion = [np.nanmean(speed, axis=1), np.nanpercentile(speed, 75, axis=1),
                  np.nanmax(speed, axis=1), (speed > 2.0).sum(1) / tracked,
                  (speed > 4.0).sum(1) / tracked]
    columns = []
    for series in motion:
        series = np.nan_to_num(series)
        now = _smooth(series, 2 * SECOND + 1)
        columns += [series, now, _shift(now, -3 * SECOND), _shift(now, 3 * SECOND),
                    _shift(now, -7 * SECOND), _shift(now, 7 * SECOND)]

    x, y = ball_xy[:, 0].astype(np.float64), ball_xy[:, 1].astype(np.float64)
    seen = ~np.isnan(x)
    ball_speed = sample_games._speed(ball_xy.astype(np.float64), half_window=3)
    with np.errstate(invalid="ignore"):
        outside = np.maximum(np.maximum(-x, x - length), np.maximum(-y, y - width))
    offsets = players[:, :, :2] - ball_xy[:, None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nearest = np.nanmin(np.hypot(offsets[:, :, 0], offsets[:, :, 1]), axis=1)
    seen_share = _smooth(seen.astype(float), 4 * SECOND + 1)
    fast_ball = _smooth(np.where(seen, (np.nan_to_num(ball_speed) > 8.0).astype(float), np.nan),
                        4 * SECOND + 1)
    columns += [seen.astype(float), seen_share, _shift(seen_share, -4 * SECOND),
                _shift(seen_share, 4 * SECOND), ball_speed, _smooth(ball_speed, 2 * SECOND + 1),
                fast_ball, outside, nearest]
    columns += _around_a_gap(outside, seen) + _shape(players, pitch)
    for series in motion[:2]:                            # a longer look either side
        wide = _smooth(np.nan_to_num(series), 6 * SECOND + 1)
        columns += [_shift(wide, -15 * SECOND), _shift(wide, 15 * SECOND)]
    return np.stack(columns, axis=1).astype(np.float32)


def _around_a_gap(outside, seen):
    """Where the ball was last seen and where it is next seen, and how long ago and ahead.

    A ball that vanished at a line went out of play; one that vanished in midfield was more
    likely lost by the tracker. Distances are to the nearest line, positive outside the pitch.
    """
    frames = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, frames, -1))
    following = np.minimum.accumulate(np.where(seen, frames, len(seen))[::-1])[::-1]
    out = []
    for index, wait in ((last, frames - last), (following, following - frames)):
        known = (index >= 0) & (index < len(seen))
        out += [np.where(known, outside[np.clip(index, 0, len(seen) - 1)], np.nan),
                np.minimum(np.where(known, wait, 3000), 3000).astype(float)]
    return out


def _shape(players, pitch):
    """How the players are arranged: set pieces have their own look (a taker on the line, a
    crowd in the box, everyone bunched or spread)."""
    length, width = pitch
    x, y = players[:, :, 0].astype(np.float64), players[:, :, 1].astype(np.float64)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        on_line = ((x < 1) | (x > length - 1) | (y < 1) | (y > width - 1)).sum(1)
        in_box = (np.abs(y - width / 2) < 20.16) & ((x < 16.5) | (x > length - 16.5))
        left, right = (in_box & (x < length / 2)).sum(1), (in_box & (x > length / 2)).sum(1)
        return [np.nanstd(x, axis=1), np.nanstd(y, axis=1), on_line.astype(float),
                np.maximum(left, right).astype(float)]


def training_set(game_numbers=sample_games.GAMES):
    inputs, answers = [], []
    for number in game_numbers:
        game = sample_games.like_ours(sample_games.load(number), seed=number)
        inputs.append(features(game["players"], game["ball_xy"],
                               sample_games.PITCH)[::TRAIN_EVERY])
        answers.append((game["labels"] == "D")[::TRAIN_EVERY])
    return np.concatenate(inputs), np.concatenate(answers)


def fit(game_numbers=sample_games.GAMES):
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=200,
                                           l2_regularization=1.0, random_state=0)
    return model.fit(*training_set(game_numbers))


def dead_evidence(model, players, ball_xy, pitch):
    """log P(dead) - log P(alive) per frame [T]."""
    p = model.predict_proba(features(players, ball_xy, pitch))[:, 1]
    return np.clip(np.log(p + 1e-6) - np.log(1 - p + 1e-6), -MAX_LLR, MAX_LLR)


def before_kickoff(n_frames, game=GAME):
    """Frames of each half before its kick-off tag starts: warm-up, never play [T] bool."""
    mask = np.zeros(n_frames, dtype=bool)
    tags = [t for t in load_smart_tagging.load(game) if t["tags"].get("Type") == "Kick Off (Start)"]
    for _, first, last in half_video_blocks(game):
        for tag in tags:
            if first < tag["start_frame"] <= last:
                mask[first:tag["start_frame"]] = True
    return mask


if __name__ == "__main__":                               # train on one game, test on the other
    for train, test in ((1, 2), (2, 1)):
        model = fit((train,))
        game = sample_games.like_ours(sample_games.load(test), seed=100 + test)
        llr = dead_evidence(model, game["players"], game["ball_xy"], sample_games.PITCH)
        truth = game["labels"] == "D"
        guess = llr > 0
        seen = ~np.isnan(game["ball_xy"][:, 0])
        print("train {} test {}: accuracy {:.3f}  dead recall {:.3f}  dead precision {:.3f}  "
              "accuracy with ball seen {:.3f}".format(
                  train, test, (guess == truth).mean(), (guess & truth).sum() / truth.sum(),
                  (guess & truth).sum() / guess.sum(), (guess == truth)[seen].mean()))
