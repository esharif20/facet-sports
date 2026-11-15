"""Restarts: where a stoppage ends, and which team restarts play.

DATA.md: "every set piece is a restart and every restart ends a dead-ball stretch". Smart Tagging
tags 59 set pieces in the competition match, most of them with the team taking it. That is free
evidence for both halves of the label: dead just before the restart, live just after, owned by
the restarting team.

Smart Tagging is automatic, though (DATA.md, "It is inferred, not ground truth"):
- its times are accurate to about a second,
- it misses restarts and invents a few,
- every throw-in has Team "N/A".
So a restart is added to the decoder's evidence, never written as a label. The frames within
MARGIN of the tagged time get no dead/live evidence, so the decoder places the boundary from the
tracking, and the weights are small enough that tracking which clearly disagrees still wins.

The weights were chosen with `python -m facet.bench restarts` on Metrica's open sample games,
whose real set pieces are first made as unreliable as Smart Tagging (like_smart_tagging below).
There it lifts spellF1 from 0.544 to 0.557. With much worse tagging (half missed, 2 s timing error,
20% invented) it neither helps nor hurts, which is why the weights are not set higher: weight 4
scores a little more with good tags but loses with bad ones. The bench is optimistic, because the
sample games' dead-ball labels are themselves cut at set pieces. Nothing here looks at the
competition ground truth.
"""

import numpy as np

from facet.load import FPS

MARGIN = 1 * FPS        # frames either side of the tagged time left to the tracking
BEFORE = 3 * FPS        # frames of dead evidence before that
AFTER = 3 * FPS         # frames of live (and team) evidence after it
DEAD_WEIGHT = 2.0       # log-likelihood ratio per frame, same units as dead.dead_evidence
TEAM_WEIGHT = 0.5       # same units as owner.team_evidence (which is clipped to +-4)

# DATA.md: the Team tag is "Barcelona" for the home side and "Atletic Club" (sic) for the away
# side. Anything else is treated as unknown rather than guessed.
HOME_TAG, AWAY_TAG = "Barcelona", "Atletic Club"

# How unreliable Smart Tagging is assumed to be, for the bench only. The competition match has 59
# tagged set pieces where a match usually has 70 to 80 restarts (the sample games have 77 and 80),
# so about a quarter are missed. Timing is "accurate to about a second" (load_smart_tagging).
MISS = 0.25
FALSE = 0.05            # extra restarts at random live frames, as a share of the real ones
JITTER = 1.0 * FPS      # standard deviation of the timing error, frames


def from_smart_tagging(game=None):
    """[(frame, side)] for every tagged set piece; side 0 home, 1 away, None unknown."""
    from games import GAME
    from starter import load_smart_tagging
    out = []
    for tag in load_smart_tagging.load(game or GAME):
        if tag["code"].strip().upper() != "SET PIECES":
            continue
        who = (tag["tags"].get("Team") or "").strip().lower()
        side = 0 if who == HOME_TAG.lower() else 1 if who == AWAY_TAG.lower() else None
        out.append((tag["start_frame"], side))
    return out


def evidence(n_frames, restarts, dead_weight=DEAD_WEIGHT, team_weight=TEAM_WEIGHT,
             margin=MARGIN, before=BEFORE, after=AFTER):
    """(team, dead) evidence [T] each, to add to owner.team_evidence and dead.dead_evidence."""
    team = np.zeros(n_frames)
    dead = np.zeros(n_frames)

    def span(a, b):
        return slice(int(np.clip(a, 0, n_frames)), int(np.clip(b, 0, n_frames)))

    for frame, side in restarts:
        dead[span(frame - margin - before, frame - margin)] += dead_weight
        dead[span(frame + margin, frame + margin + after)] -= dead_weight
        if side is not None:
            team[span(frame - margin, frame + margin + after)] += team_weight * (1 - 2 * side)
    return team, dead


def like_smart_tagging(restarts, labels, seed, miss=MISS, false=FALSE, jitter=JITTER):
    """A copy of a sample game's true set pieces, made as unreliable as Smart Tagging: some
    dropped, some invented, times shifted, and throw-ins without a team.

    restarts: [(frame, side, kind)] from sample_games.load. labels: that game's labels [T]."""
    rng = np.random.default_rng(seed)
    out = []
    for frame, side, kind in restarts:
        if rng.random() < miss:
            continue
        shifted = int(round(frame + rng.normal(0, jitter)))
        out.append((shifted, None if "THROW" in kind else side))
    live = np.where(labels != "D")[0]
    for _ in range(int(round(false * len(restarts)))):
        out.append((int(rng.choice(live)), int(rng.integers(0, 2))))
    return out
