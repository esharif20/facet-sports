# Facet Sports — technical document

More than a Hack 2027, Phase 1: ball possession for BAR vs ATH.

## Approach

We label each frame as home possession (H), away possession (A) or dead ball (D) in five steps.

1. **One ball track from two cameras.** The broadcast tracking is synchronised to the tactical
   tracking by sliding one ball track along the other (the offset of each half is found by the
   code). The tactical ball is used where it exists and the broadcast ball fills its gaps. Where
   the two feeds disagree by more than 3 m the ball is marked "disputed" and its position is not
   trusted: either it is in the air, or one tracker follows something else.
2. **Two classifiers per frame**, both gradient-boosted decision trees
   (`HistGradientBoostingClassifier`, 400 trees each):
   - *Dead ball:* is play stopped? Inputs: how fast the players move, whether the ball is seen,
     whether it left the pitch, where it was last seen and next seen, how the players are
     arranged, and the same quantities up to 15 s before and after.
   - *Ownership:* which team has the ball? Inputs: which team's player is nearest the ball, the
     stretches each team spends in contact with it, touches (sudden changes of ball velocity next
     to a player), which way play drifts, and how the two teams sit (territory, spread and speed
     of each team). The last group decides ownership while the ball cannot be seen.
3. **A second look at ownership.** A third classifier of the same kind reads the first two's
   answers up to 7 s either side of the frame, and how long each run of answers for the same team
   lasts, and corrects the frame's ownership. It learns, for example, that a second of weak
   evidence for one team between two long spells of the other is usually not a possession. It is
   trained on answers the first two gave for matches they had not seen (five folds over the
   training matches), so it learns from the mistakes they make on a new match.
4. **A decoder.** A three-state Viterbi (H, A, D) finds the best sequence of states for each
   half. Changing state costs a fixed penalty, so a change only happens when the evidence after
   it, summed over frames, outweighs the penalty. This turns flickering per-frame probabilities
   into whole possessions, which is what `spellF1` rewards.
5. **No X on graded frames.** Every frame of both halves gets H, A or D.

**Why learned models rather than rules.** Our first version was hand rules (nearest player with
hysteresis, dead ball when the ball is missing and players walk). It scored 0.41 `spellF1` on the
public window. Rules could not express when a touch counts as control or when a visible ball is
still dead. Both are learned easily from matches where the answer is known.

**Why train on other matches.** The rules forbid labels from the provided game, so the models
are trained on ten other matches (see Training data). Their tracking is clean and ours is
automatic, so each training match is first roughened until its ball looks like ours: positions
are smoothed, a slowly wandering error of 0.8 m is added, the ball is hidden on a quarter of live
frames in stretches of 3 to 15 s, and a visible ball is drawn into some stoppages. These settings
were chosen by matching statistics of our tracking that need no labels (distance from the ball to
the nearest player, ball acceleration, how often the ball is seen). Each training match is
roughened twice with different noise.

**How settings were chosen.** Always on training matches scored with the official metric by
models that never saw them, never on the provided game. `python -m facet.bench` does this for
Metrica's three sample games, and features and model sizes were kept or dropped there. The
decoder penalties were set on the same kind of check over all ten training matches.

In a leave-one-match-out check over all ten training matches, run during development, the
pipeline scores `spellF1` 0.65 and `macroF1` 0.91 against labels taken exactly as the source
names them. The second look raised `spellF1` on every one of the ten. Training on unsmoothed
labels instead of the 2 s rule was worth 0.04 in that check. Scored instead against labels
smoothed with the 2 s rule, the submitted pipeline reaches 0.716 against 0.719 for one trained
on smoothed labels, so the choice costs little if the challenge's labels are smoother than ours.

## Assets used, and dropped

| Asset | Use |
|---|---|
| Tactical tracking | Players and ball. The main input. |
| Broadcast tracking | Ball only, synchronised to tactical frames. Fills gaps and flags disputed frames. |
| Smart Tagging | One tag only: "Kick Off (Start)". Frames of a half before it are dead. |
| Homography, videos | Not used in the submitted pipeline (see below). |

Tried and dropped:

- **Ball height from two cameras.** Intersecting the two cameras' sight lines gave heights that
  looked plausible, but checking video frames showed the large heights were a tracker following
  the wrong object, not a ball in the air. Replaced by the disputed flag.
- **Our own ball detector.** A small convolutional network (6 layers, about 110,000 weights),
  trained with no human labels on image patches where both trackers agree on the ball. It found
  the ball on held-out patches (recall 0.93, precision 0.91) and raised the share of frames with a
  usable ball from 54% to 70%. It did not improve the labels: the trackers already see the ball
  on 94% of live frames, and two thirds of the new ball frames fall in stoppages. A bench test
  with the hidden-ball share lowered to match showed a ceiling of +0.009 `spellF1`. It also took
  100 minutes to run and would have made an identical rerun harder to guarantee.
- **Smart Tagging as a possession hint** (team-labelled phases, set-piece team as restart team).
  No gain where it could be measured, and the tag times do not coincide with the restart.
- **A change-of-possession classifier** that lowers the decoder's penalty at likely turnovers:
  +0.009 on the bench, within noise.
- **Published possession statistics for this match.** Not used: human-produced data about the
  provided game is forbidden.

## Training data

No label of any kind comes from the provided game. Its public 5-minute ground truth was used
only to check that the pipeline runs and that labels mean what we think.

| Dataset | Matches | Source | Labels |
|---|---|---|---|
| Metrica Sports sample data, games 1 to 3 | 3 | https://github.com/metrica-sports/sample-data | Tracking and event data published by Metrica. |
| Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer" | 7 (Bundesliga and 2. Bundesliga, 2022/23) | https://doi.org/10.6084/m9.figshare.28196177, CC BY | Official DFL tracking, which carries on every frame the team in possession and whether the ball is in play. |

About 1.4 million frames in total. Nobody in our team labelled anything. Our code turns the
published data into H, A, D:

- **Metrica games:** D where the hand-corrected ball is absent for at least 1 s; otherwise the
  team of the last event that names an owner (set piece, pass, carry, shot, recovery, ball lost).
- **Bundesliga matches:** D where the ball-in-play flag is off for at least 1 s; otherwise the
  possession flag.
- **Both:** nothing is smoothed afterwards. An earlier version gave any spell shorter than 2 s
  between two spells of the other team back to that team, as our reading of "controlled
  possession". The public window, which the challenge names as the reference, contains such
  spells of 0.8 s and 1.3 s, so the rule was removed.

## Possession rules implemented

The rules are learned from the training labels above, so they are the rules of those labels:

- **Change of possession** happens where the training labels put it: at the first event that
  names the other team as owner (Metrica) or where the official possession flag changes
  (Bundesliga). A deflection or a challenge the first team wins is not such an event. Brief
  possessions count, down to under a second. The decoder's penalty for changing team keeps a
  single odd frame from becoming one.
- **Loose balls and passes in flight** stay with the team that last had the ball until the other
  team controls it.
- **Dead ball** runs from the whistle or the ball leaving the pitch until the restart is taken,
  even when the ball is visible and being carried or placed. The time before each half's kickoff
  is dead.
- **Restarts** belong to the team that first controls the ball after the restart.

## Limitations

- **Short possessions.** We recover most possessions longer than 5 s and fewer than half of
  those under 5 s. A short spell between two long ones gives little evidence, and missing it
  also breaks the match of the two spells around it.
- **Short stoppages.** On the training matches we find 63% of stoppages under 5 s, 86% of those
  between 5 and 10 s and nearly all longer ones.
- **Hidden or disputed ball.** Ownership accuracy on the bench is about 0.94 with the ball seen
  and 0.87 without. Team shape helps but does not replace the ball.
- **Ball in the air.** The homography places an airborne ball at the wrong ground position. We
  only flag those frames; we do not reconstruct the flight.
- **Training matches differ from ours.** Their labelling conventions are not exactly Metrica's
  for this challenge, and the roughening is a simulation of our tracking errors, not the errors
  themselves.
- **With more time:** a decoder that knows typical possession and stoppage lengths, a neural
  sequence model in place of the second classifier, and a ball track rebuilt in 3D.

## Reproduction

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
# unpack the match data into data/tactical/ and data/broadcast/ (see DATA.md)
.venv/bin/python -m facet.run
```

This writes `evaluation/results/facet.csv`, validates it and scores it on the public window. The
first run downloads the training matches (about 2.9 GB) into `data/external/`. A run takes about
half an hour, most of it training. Nothing is random at run time: the noise used to roughen the training matches has fixed
seeds, the classifiers have a fixed seed, and all versions are pinned in `requirements.lock`.

`python -m facet.bench` reproduces the bench numbers, and `python -m pytest tests -q` runs the
tests.
