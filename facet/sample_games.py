"""Other matches with known possession, used to train and tune. None is the competition match,
and nothing here touches its ground truth.

Metrica's open sample games 1, 2 and 3 (https://github.com/metrica-sports/sample-data): labels
from Metrica's own event data. The bench scores on these.

Labels are taken exactly as the source names them, with nothing smoothed, because that is how
the challenge says its ground truth is made. Its public window shows possessions under a second
between two spells of the other team, and so do these matches.

Seven Bundesliga matches of the 2022/23 season (Bassek et al., "An integrated dataset of
spatiotemporal and event data in elite soccer", https://doi.org/10.6084/m9.figshare.28196177,
CC BY): the official tracking carries, on every frame, which team has the ball and whether it is
in play. Used for training only.

    python -m facet.sample_games     downloads the files into data/external/ (about 2.9 GB)
"""

import io
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from facet.load import CACHE_DIR, FPS
from games import DATA_DIR

FOLDER = os.path.join(DATA_DIR, "external", "metrica")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
GAMES = (1, 2, 3)
FILES = {1: ("RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"),
         2: ("RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"),
         3: ("events.json", "metadata.xml", "tracking.txt")}
DFL_FOLDER = os.path.join(DATA_DIR, "external", "dfl")
DFL_URL = "https://ndownloader.figshare.com/files/"
DFL_FILES = {"J03WMX": (51643475, 51643514), "J03WN1": (51643472, 51643517),     # match: its
             "J03WOH": (51643478, 51643520), "J03WOY": (51643481, 51643523),     # information and
             "J03WPY": (51643487, 51643526), "J03WQQ": (51643484, 51643529),     # positions files
             "J03WR9": (51643490, 51643532)}
DFL = tuple(DFL_FILES)
TRAINING = GAMES + DFL
DRAWS = 2               # times each game is roughened for training, each with different noise
PITCH = (105.0, 68.0)   # of every game here
OWNING = ("SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST")   # events naming the owner

# The sample games are hand-corrected: their ball is glued to whoever has it and never lost in
# play. Ours is automatic. These settings damage the sample ball until it looks like ours. They
# were set by matching two things measured on our tracking with no labels involved: how far the
# nearest player is from the ball (median 1.65 m) and how smoothly the ball moves.
SMOOTH = 9              # frames: our tracker rounds off sharp changes of direction
WANDER = 0.8            # m: how far the tracked ball strays from the real one
WANDER_FRAMES = 25      # how slowly that error changes
HIDE_SHARE = 0.25       # of live frames lose their ball, as our tracker does in play
HIDE_SECONDS = (3, 15)
# Our tracker often still sees the ball while it is dead: rolling on after the whistle, carried
# back, or placed for the restart. The sample games never show that, so it is drawn in. A quick
# restart (throw-in, goal kick) keeps the ball in view; a long stoppage (goal, injury) loses it.
# Chances of: nothing shown, the roll-on only, the placed ball only, the whole stoppage.
DEAD_BALL_QUICK = (0.15, 0.10, 0.25, 0.50)
DEAD_BALL_LONG = (0.50, 0.15, 0.25, 0.10)
QUICK = 20              # s: stoppages shorter than this are quick restarts
ROLL_ON = 1.5           # s: how quickly a ball left alone slows down


def _path(game, name):
    return os.path.join(FOLDER, "Sample_Game_{0}_{1}".format(game, name))


def download():
    os.makedirs(FOLDER, exist_ok=True)
    for game, names in FILES.items():
        for name in names:
            if not os.path.exists(_path(game, name)):
                urllib.request.urlretrieve("{0}Sample_Game_{1}/Sample_Game_{1}_{2}".format(
                    URL, game, name), _path(game, name))
    os.makedirs(DFL_FOLDER, exist_ok=True)
    for match, numbers in DFL_FILES.items():
        for kind, number in zip(("info", "positions"), numbers):
            path = os.path.join(DFL_FOLDER, "{}_{}.xml".format(kind, match))
            if not os.path.exists(path):
                urllib.request.urlretrieve(DFL_URL + str(number), path + ".part")
                os.rename(path + ".part", path)          # a broken download leaves no file behind


def _speed(xy, half_window=12):
    """Speed in m/s over about a second, which rides over frame-to-frame jitter."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    step = xy[2 * half_window:] - xy[:-2 * half_window]
    out[half_window:-half_window] = np.hypot(step[..., 0], step[..., 1]) * FPS / (2 * half_window)
    return out


def with_speed(xy):
    """[T,n,2] positions -> [T,n,3] with speed, the layout facet.load uses."""
    return np.concatenate([xy, _speed(xy)[..., None]], axis=-1).astype(np.float32)


def _csv_game(game):
    """Games 1 and 2: (xy [T,n,2] in 0-1, team [n], ball [T,2] in 0-1, period [T], events)."""
    sides = [pd.read_csv(_path(game, "RawTrackingData_{}_Team.csv".format(side)), skiprows=3,
                         header=None).to_numpy(np.float64) for side in ("Home", "Away")]
    home, away = sides[0][:, 3:-2], sides[1][:, 3:-2]
    xy = np.concatenate([home, away], axis=1).reshape(len(home), -1, 2)
    team = np.r_[np.zeros(home.shape[1] // 2, np.int8), np.ones(away.shape[1] // 2, np.int8)]
    table = pd.read_csv(_path(game, "RawEventsData.csv"))
    events = [(row[0] == "Home", row[1], str(row[2]), int(row[4]), int(row[6]))
              for row in table.itertuples(index=False)]
    return xy, team, sides[0][:, -2:], sides[0][:, 0].astype(int), events


def _epts_game(game):
    """Game 3, in the same text format as the competition tracking. Which player each column
    holds changes with substitutions, so every player gets a column of their own."""
    meta = open(_path(game, "metadata.xml")).read()
    team_of = dict(re.findall(r'<Player id="([^"]+)" teamId="([^"]+)"', meta))
    player_of = dict(re.findall(r'<PlayerChannel channelId="x" id="([^"]+)" playerId="([^"]+)"',
                                meta))
    if not player_of:                                    # attribute order differs between files
        player_of = {c: p for p, c in re.findall(
            r'<PlayerChannel[^>]*playerId="([^"]+)"[^>]*id="([^"]+_x)"', meta)}
    ids = sorted(team_of)
    home_team = re.search(r'<Team id="([^"]+)"', meta).group(1)
    team = np.array([0 if team_of[i] == home_team else 1 for i in ids], dtype=np.int8)

    text = open(_path(game, "tracking.txt")).read().replace(":", ",").replace(";", ",")
    rows = pd.read_csv(io.StringIO(text), header=None).to_numpy(np.float64)
    xy = np.full((len(rows), len(ids), 2), np.nan)
    for first, last, body in re.findall(
            r'<DataFormatSpecification endFrame="(\d+)" separator=":" startFrame="(\d+)">(.*?)'
            r'</DataFormatSpecification>', meta, re.S):
        first, last = int(last), int(first)              # attributes come as end, then start
        channels = re.findall(r'playerChannelId="([^"]+_x)"', body)
        block = (rows[:, 0] >= first) & (rows[:, 0] <= last)
        for column, channel in enumerate(channels):
            slot = ids.index(player_of[channel])
            xy[block, slot] = rows[block, 1 + 2 * column:3 + 2 * column]
    half = int(re.search(r"<Name>second_half_start</Name>\s*<Value>(\d+)", meta).group(1))
    period = np.where(rows[:, 0] >= half, 2, 1)

    events = []
    for event in json.load(open(_path(game, "events.json")))["data"]:
        subtypes = event["subtypes"] or []
        subtypes = [subtypes] if isinstance(subtypes, dict) else subtypes
        events.append((event["team"]["id"] == home_team, event["type"]["name"],
                       "-".join(sub["name"] for sub in subtypes),
                       int(event["start"]["frame"]), int(event["end"]["frame"])))
    return xy, team, rows[:, -2:], period, events


def _dfl_game(match):
    """One Bundesliga match, in the layout load() returns. Parsing 400 MB of XML takes a while,
    so the arrays are kept in data/cache/."""
    cache = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match))
    if not os.path.exists(cache):
        info = open(os.path.join(DFL_FOLDER, "info_{}.xml".format(match))).read()
        home = re.search(r'HomeTeamId="([^"]+)"', info).group(1)
        text = open(os.path.join(DFL_FOLDER, "positions_{}.xml".format(match))).read()
        tracks = {}                                       # (half, team, person) -> its frames
        for half, team, person, body in re.findall(
                r'<FrameSet GameSection="(firstHalf|secondHalf)"[^>]*TeamId="([^"]+)" '
                r'PersonId="([^"]+)">(.*?)</FrameSet>', text, re.S):
            if team != "referee":
                flags = r'[^>]*?BallPossession="(\d)" BallStatus="(\d)"' if team == "BALL" else ""
                tracks[half, team, person] = np.array(re.findall(
                    r'N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"' + flags, body), np.float64)
        people = sorted({key[1:] for key in tracks if key[1] != "BALL"})
        parts = []
        for period, half in ((1, "firstHalf"), (2, "secondHalf")):
            ball = next(rows for key, rows in tracks.items() if key[:2] == (half, "BALL"))
            first, n = int(ball[0, 0]), int(ball[-1, 0] - ball[0, 0]) + 1
            xy, flags = np.full((n, len(people) + 1, 2), np.nan), np.zeros((n, 2), np.int8)
            flags[ball[:, 0].astype(int) - first] = ball[:, 3:]
            for (its_half, team, person), rows in tracks.items():
                if its_half == half:
                    at = rows[:, 0].astype(int) - first
                    ok = (at >= 0) & (at < n)
                    slot = len(people) if team == "BALL" else people.index((team, person))
                    xy[at[ok], slot] = rows[ok, 1:3]
            parts.append((xy, flags, np.full(n, period, np.int8)))
        xy, flags, period = (np.concatenate(column) for column in zip(*parts))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(                              # the origin is the centre spot
            cache, xy=(xy + np.array(PITCH) / 2).astype(np.float32), period=period,
            team=np.array([0 if team == home else 1 for team, _ in people], np.int8),
            home_has_it=flags[:, 0] == 1, in_play=flags[:, 1] == 1)
    saved = np.load(cache)
    dead = _stoppages(~saved["in_play"])
    ball = saved["xy"][:, -1].astype(np.float64)
    ball[dead] = np.nan                                   # as in Metrica's games: no ball while dead
    return {"players": with_speed(saved["xy"][:, :-1].astype(np.float64)), "ball_xy": ball,
            "team": saved["team"], "period": saved["period"],
            "labels": np.where(dead, "D", np.where(saved["home_has_it"], "H", "A"))}


def _stoppages(dead):
    """Dead frames, without the runs under a second: those are dropouts, not stoppages."""
    dead = dead.copy()
    edges = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if end - start < FPS:
            dead[start:end] = False
    return dead


def load(game):
    """dict: players [T,n,3], ball_xy [T,2] (metres), team [n], labels [T] of 'H' 'A' 'D',
    period [T]."""
    download()
    if game in DFL:
        return _dfl_game(game)
    xy, team, ball, period, events = (_epts_game if game == 3 else _csv_game)(game)
    n = len(xy)
    # In hand-corrected tracking the ball is absent exactly while it is dead: it vanishes one
    # frame after the stopping event and returns at the restart. Shorter holes are dropouts.
    dead = _stoppages(np.isnan(ball[:, 0]))
    owner = np.full(n + 1, -1, dtype=np.int8)
    for is_home, kind, _, start, _ in sorted(events, key=lambda e: e[3]):
        if kind in OWNING:
            owner[start] = 0 if is_home else 1
    for frame in range(1, n + 1):                        # the owner keeps the ball until the next
        if owner[frame] < 0:
            owner[frame] = owner[frame - 1]
    labels = np.where(dead, "D", np.where(owner[1:] == 0, "H", "A"))
    return {"players": with_speed(xy * PITCH), "ball_xy": ball * PITCH, "team": team,
            "labels": labels, "period": period}


def _smooth(values, frames):
    """Mean over a centred window, ignoring NaN; NaN stays NaN."""
    ok = ~np.isnan(values)
    kernel = np.ones(frames)
    total = np.convolve(np.where(ok, values, 0.0), kernel, mode="same")
    return np.where(ok, total / np.maximum(np.convolve(ok.astype(float), kernel, mode="same"), 1),
                    np.nan)


def _dead_ball(ball, labels, rng):
    """A copy of the ball with sightings drawn into some stoppages (see DEAD_BALL_QUICK)."""
    out = ball.copy()
    edges = np.diff(np.r_[0, (labels == "D").astype(np.int8), 0])
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if start < 4 or end >= len(ball) or np.isnan(ball[[start - 4, start - 1, end], 0]).any():
            continue
        kind = rng.choice(4, p=DEAD_BALL_QUICK if end - start < QUICK * FPS else DEAD_BALL_LONG)
        if kind == 0:
            continue
        leaving, restart = ball[start - 1], ball[end]
        velocity = (ball[start - 1] - ball[start - 4]) * FPS / 3
        roll = min(int(rng.integers(1 * FPS, 6 * FPS)), end - start)
        t = np.arange(1, roll + 1)[:, None] / FPS
        rolled = leaving + velocity * ROLL_ON * (1 - np.exp(-t / ROLL_ON))
        placed = min(int(rng.integers(1 * FPS, 10 * FPS)), end - start)
        if kind in (1, 3):
            out[start:start + roll] = rolled
        if kind in (2, 3):
            out[end - placed:end] = restart
        if kind == 3 and start + roll < end - placed:        # carried from one to the other
            steps = np.linspace(0, 1, end - placed - start - roll + 2)[1:-1, None]
            out[start + roll:end - placed] = rolled[-1] + steps * (restart - rolled[-1])
    return out


def like_ours(game, seed):
    """A copy of a game whose ball is as rough as our tracking: seen during some stoppages,
    rounded off, wandering, and missing on random stretches of live play."""
    rng = np.random.default_rng(seed)
    n = len(game["ball_xy"])
    window = np.hanning(2 * WANDER_FRAMES + 1)
    window /= np.sqrt((window ** 2).sum())
    white = rng.normal(0, WANDER, (n + 4 * WANDER_FRAMES, 2))
    wander = np.stack([np.convolve(white[:, axis], window, mode="same") for axis in (0, 1)], 1)
    drawn = _dead_ball(game["ball_xy"], game["labels"], rng)
    ball = np.stack([_smooth(drawn[:, axis], SMOOTH) for axis in (0, 1)], 1) \
        + wander[2 * WANDER_FRAMES:2 * WANDER_FRAMES + n]

    live = np.where(game["labels"] != "D")[0]
    hidden = 0
    while hidden < HIDE_SHARE * len(live):
        start = rng.choice(live)
        length = rng.integers(HIDE_SECONDS[0] * FPS, HIDE_SECONDS[1] * FPS)
        ball[start:start + length] = np.nan
        hidden += length
    return dict(game, ball_xy=ball)


def training_copies(game):
    """One game roughened DRAWS times. Different noise each time teaches the models what is noise."""
    data = load(game)
    return [like_ours(data, seed=TRAINING.index(game) + 1 + 1000 * draw) for draw in range(DRAWS)]


if __name__ == "__main__":
    for number in TRAINING:
        game = load(number)
        labels = game["labels"]
        starts = np.r_[True, labels[1:] != labels[:-1]]
        spells = int((starts & (labels != "D")).sum())
        print("game {}: {} frames, dead {:.1%}, {} possessions of mean {:.1f} s, ball seen "
              "{:.1%}".format(number, len(labels), (labels == "D").mean(), spells,
                              (labels != "D").sum() / spells / FPS,
                              (~np.isnan(game["ball_xy"][:, 0])).mean()))
