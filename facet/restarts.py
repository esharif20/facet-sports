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
There it lifts spellF1 from 0.554 to 0.564 (macroF1 0.772 to 0.778). With much worse tagging (half missed, 2 s timing error,
20% invented) it neither helps nor hurts, which is why the weights are not set higher: weight 4
scores a little more with good tags but loses with bad ones. The bench is optimistic, because the
sample games' dead-ball labels are themselves cut at set pieces. Nothing here looks at the
competition ground truth.

Two additions on top (facet.bench prints each one's effect separately):
- Throw-ins: their side is read from the tracking (throw_in_side), as the team of the player on
  the touchline around the restart. On the sample games' 110 throw-ins it names a side for 97% and
  is right on 99% of those; their player tracking is cleaner than ours, so expect less here.
- Goals: the stretch from a GOALS tag to the "Kick Off (After Goal)" after it is forced to dead,
  as the warm-up is (goal_stoppages). A goal is never missed by the scoreboard, and DATA.md counts
  4 of each tag, so this one is trusted further than the other set pieces.
Together they take spellF1 from 0.564 to 0.566 and macroF1 from 0.778 to 0.779 on the bench. That
is small: on the sample games the dead-ball model already finds most celebrations, and the ownership
model most throw-in teams. Both should matter more on our rougher tracking.
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


# Throw-ins: Smart Tagging never names the team, so it is read from the tracking. The thrower
# stands on or just behind a touchline while the other players are on the pitch, so the player
# nearest the line (or furthest past it) around the restart is taken to be the thrower.
LINE = 1.0              # m: a player this close to a touchline, or past it, can be the thrower
OFF_PITCH = 4.0         # m: further behind the line than this is a substitute or staff, not play
THROW_WINDOW = (-2 * FPS, 2 * FPS)   # frames around the tagged time; tags are ~1 s off
MIN_VOTES = 10          # frames with someone on the line, below which the side stays unknown
MIN_SHARE = 0.7         # share of those frames one team must hold

# Goals: from the goal to the kick-off after it is one long stoppage (celebration, walk back).
# The sample games' gaps are 53 to 75 s. The frames between are forced to dead, like the warm-up.
AFTER_GOAL = 3 * FPS    # frames after the GOALS tag ends before forcing, in case it ends early
BEFORE_KICKOFF = 2 * FPS   # frames before the kick-off tag left to the tracking
NO_GOAL_TAG = 40 * FPS  # without a GOALS tag to pair with, force only this long before kick-off
MAX_STOPPAGE = 150 * FPS   # a GOALS tag further back than this is not this kick-off's goal


def _side(tag):
    who = (tag["tags"].get("Team") or "").strip().lower()
    return 0 if who == HOME_TAG.lower() else 1 if who == AWAY_TAG.lower() else None


def from_smart_tagging(game=None):
    """[(frame, side, kind)] for every tagged set piece; side 0 home, 1 away, None unknown;
    kind is the Type tag, upper case ("THROW-IN", "CORNER KICK", ...)."""
    from games import GAME
    from starter import load_smart_tagging
    return [(tag["start_frame"], _side(tag), (tag["tags"].get("Type") or "").strip().upper())
            for tag in load_smart_tagging.load(game or GAME)
            if tag["code"].strip().upper() == "SET PIECES"]


def is_throw_in(kind):
    return "THROW" in (kind or "").upper()


def throw_in_side(players, team, frame, pitch, window=THROW_WINDOW):
    """0 home, 1 away, or None: the team of the player on the touchline around `frame`.

    players [T,n,>=2] in metres, team [n]. Each frame in the window votes for the team of the
    player closest to (or furthest past) a touchline, if that player is within LINE of it.
    """
    lo, hi = max(frame + window[0], 0), min(frame + window[1], len(players))
    if hi <= lo:
        return None
    y = players[lo:hi, :, 1].astype(np.float64)
    to_line = np.minimum(y, pitch[1] - y)               # negative behind the line
    to_line = np.where(np.isnan(to_line) | (to_line < -OFF_PITCH), np.inf, to_line)
    nearest = to_line.argmin(1)
    close = to_line[np.arange(len(y)), nearest] < LINE
    votes = team[nearest[close]]
    votes = votes[(votes == 0) | (votes == 1)]
    if len(votes) < MIN_VOTES:
        return None
    home = (votes == 0).mean()
    return 0 if home >= MIN_SHARE else 1 if home <= 1 - MIN_SHARE else None


def with_throw_in_sides(restarts, players, team, pitch):
    """The restarts, with each throw-in that has no team given the side read from tracking."""
    return [(frame, throw_in_side(players, team, frame, pitch)
             if side is None and is_throw_in(kind) else side, kind)
            for frame, side, kind in restarts]


def goals_from_tags(tags):
    """[(goal_end_frame or None, kickoff_frame)] from Smart Tagging rows: each "Kick Off (After
    Goal)" set piece, paired with the last GOALS tag that ends before it, if one is close enough."""
    goals = sorted(tag["end_frame"] for tag in tags if tag["code"].strip().upper() == "GOALS")
    out = []
    for tag in tags:
        if (tag["code"].strip().upper() == "SET PIECES"
                and (tag["tags"].get("Type") or "").strip().lower() == "kick off (after goal)"):
            kickoff = tag["start_frame"]
            before = [g for g in goals if kickoff - MAX_STOPPAGE <= g < kickoff]
            out.append((before[-1] if before else None, kickoff))
    return out


def goals_from_smart_tagging(game=None):
    from games import GAME
    from starter import load_smart_tagging
    return goals_from_tags(load_smart_tagging.load(game or GAME))


def goal_stoppages(n_frames, goals):
    """[T] bool: frames certainly dead between a goal and the kick-off after it."""
    mask = np.zeros(n_frames, dtype=bool)
    for goal, kickoff in goals:
        first = kickoff - NO_GOAL_TAG if goal is None else goal + AFTER_GOAL
        first = max(first, kickoff - MAX_STOPPAGE)
        last = kickoff - BEFORE_KICKOFF
        mask[int(np.clip(first, 0, n_frames)):int(np.clip(last, 0, n_frames))] = True
    return mask


def evidence(n_frames, restarts, dead_weight=DEAD_WEIGHT, team_weight=TEAM_WEIGHT,
             margin=MARGIN, before=BEFORE, after=AFTER):
    """(team, dead) evidence [T] each, to add to owner.team_evidence and dead.dead_evidence."""
    team = np.zeros(n_frames)
    dead = np.zeros(n_frames)

    def span(a, b):
        return slice(int(np.clip(a, 0, n_frames)), int(np.clip(b, 0, n_frames)))

    for frame, side, *_ in restarts:
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
        out.append((shifted, None if is_throw_in(kind) else side, kind))
    live = np.where(labels != "D")[0]
    for _ in range(int(round(false * len(restarts)))):
        out.append((int(rng.choice(live)), int(rng.integers(0, 2)), "INVENTED"))
    return out


def goals_like_smart_tagging(goals, seed, jitter=JITTER):
    """A sample game's true (goal, kick-off) frames, both shifted by Smart Tagging's timing error.
    None are dropped: DATA.md counts 4 GOALS tags and 4 kick-offs after a goal, a perfect match."""
    rng = np.random.default_rng(seed)
    return [(int(round(goal + rng.normal(0, jitter))), int(round(kickoff + rng.normal(0, jitter))))
            for goal, kickoff in goals]
