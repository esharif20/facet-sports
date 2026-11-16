# Facet Sports technical document

More than a Hack 2027, Phase 1. Ball possession for BAR vs ATH.

## Approach

Every frame gets one of three labels: home possession (H), away possession (A) or dead ball (D).
We never output X on a graded frame. The pipeline has four parts.

**One ball track from two cameras.** We synchronise the broadcast tracking to the tactical
tracking by sliding one ball track along the other. The code finds the offset for each half, so
nothing is typed in. The tactical ball is used where it exists and the broadcast ball fills its
gaps. Where the two feeds put the ball more than 3 m apart we mark it "disputed" and stop
trusting its position. Either the ball is in the air or one tracker is following something else,
and we cannot tell which.

**Two classifiers that judge each frame.** Both are gradient-boosted decision trees
(scikit-learn's `HistGradientBoostingClassifier`, 400 trees each). The dead-ball classifier
answers "is play stopped?". It looks at how fast the players move, whether the ball is seen,
whether it left the pitch, where it was last seen and next seen, how the players are arranged,
and all of that again up to 15 s before and after. The ownership classifier answers "which team
has the ball?". It looks at which team's player is nearest the ball, the stretches each team
spends in contact with it, touches (a sudden change of ball velocity next to a player), which way
play is drifting, and how the two teams sit on the pitch. That last group matters most when the
ball cannot be seen.

**A second look at ownership.** A third classifier of the same kind reads the first two's answers
up to 7 s either side of the frame, plus how long each run of answers for one team lasts, and
corrects the frame's ownership. It learns things like "a second of weak evidence for one team
between two long spells of the other is usually not a possession". We train it on answers the
first two gave for matches they had not seen, using five folds over the training matches, so what
it learns is the mistakes they make on a new match.

**A decoder.** A three-state Viterbi (H, A, D) picks the best sequence of states for each half.
Changing state costs a fixed penalty, so the label only changes when the evidence after the
change, added up over frames, beats the penalty. This is what turns flickering per-frame
probabilities into whole possessions, and whole possessions are what `spellF1` counts.

### Why learned models and not rules

Our first version was hand rules: nearest player with hysteresis, and dead ball when the ball is
missing and the players walk. It scored 0.41 `spellF1` on the public window. We could not write a
rule for when a touch counts as control, or for when a ball that is plainly visible is still
dead. A model picks both up from matches where the answer is known.

### Why train on other matches

The rules forbid labels from the provided game, so we train on ten other matches (listed under
Training data). Their tracking is clean and ours is automatic. A model trained on clean tracking
does badly on ours, so we first damage each training match until its ball looks like our ball. We
smooth the positions, add a slowly wandering error of 0.8 m, hide the ball on a quarter of live
frames in stretches of 3 to 15 s, and draw a visible ball into some of the stoppages. We set those
numbers by matching statistics of our own tracking that need no labels, such as the distance from
the ball to the nearest player and how often the ball is seen. Each training match is damaged
twice with different noise.

### How we chose settings

Never on the provided game. Every choice was made on training matches, scored with the official
metric by models that had not seen them. `python -m facet.bench` runs that check on Metrica's
three sample games, and it is where we kept or dropped features. The decoder penalties came from
the same kind of check run over all ten training matches.

That ten-match check was a development script and is not in the repository. In it the submitted
pipeline scores `spellF1` 0.65 and `macroF1` 0.91 against labels taken exactly as the source
names them. The second look raised `spellF1` on all ten matches.

We do not expect the same number on the provided game. The training matches only imitate our
tracking errors, and the public window has 17 possessions, which is too few to check a score
against.

## Assets we used and dropped

| Asset | Use |
|---|---|
| Tactical tracking | Players and ball. The main input. |
| Broadcast tracking | Ball only, synchronised to tactical frames. Fills gaps and flags disputed frames. |
| Smart Tagging | One tag, "Kick Off (Start)". Frames of a half before it are dead. |
| Homography, videos | Not used in the submitted pipeline. See below. |

Things we tried and dropped:

**Ball height from two cameras.** We intersected the two cameras' sight lines to get the ball's
height, and the numbers looked right. They were not. When we pulled the video frames, the tall
"flights" were a tracker sitting on the wrong object while the ball rolled along the ground
somewhere else. We replaced the height with the disputed flag.

**Our own ball detector.** A small convolutional network, 6 layers and about 110,000 weights,
trained without human labels on image patches where both trackers agree on the ball. It worked as
a detector (recall 0.93 and precision 0.91 on held-out patches) and raised the share of frames
with a usable ball from 54% to 70%. It did not improve the labels. The trackers already see the
ball on 94% of live frames, and two thirds of the detector's new ball frames fall in stoppages,
where ownership does not matter. A bench test with fewer hidden-ball frames put the ceiling at
+0.009 `spellF1`. It also took 100 minutes to run and would have made an identical rerun harder
to guarantee, so it is not in the submitted pipeline.

**Smart Tagging as a possession hint.** We tried the team-labelled phases as a prior and the
set-piece team as the restart team. Neither helped where we could measure it, and the tag times
do not line up with the restart.

**A change-of-possession classifier** that lowers the decoder's penalty at likely turnovers. It
gained 0.009 on the bench, which is inside the noise.

**A decoder with duration priors** (semi-Markov). It was worse than the plain Viterbi on the
training matches we tried it on.

**Published possession statistics for this match.** A teammate found them. We did not use them,
because human-produced data about the provided game is forbidden.

## Training data

No label of any kind comes from the provided game. We used its public 5-minute ground truth for
two things only: to check that the pipeline runs, and to check that our labels mean what the
challenge's labels mean.

| Dataset | Matches | Source | Labels |
|---|---|---|---|
| Metrica Sports sample data, games 1 to 3 | 3 | https://github.com/metrica-sports/sample-data | Tracking and event data published by Metrica. |
| Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer" | 7 (Bundesliga and 2. Bundesliga, 2022/23) | https://doi.org/10.6084/m9.figshare.28196177, CC BY | Official DFL tracking, which carries on every frame the team in possession and whether the ball is in play. |

That is about 1.4 million frames. Nobody in our team labelled anything. Our code turns the
published data into H, A and D like this.

For the Metrica games, a frame is D where the hand-corrected ball is absent for at least a
second. Otherwise it belongs to the team of the last event that names an owner (set piece, pass,
carry, shot, recovery, ball lost).

For the Bundesliga matches, a frame is D where the official ball-in-play flag is off, and
otherwise it takes the official possession flag.

Nothing is smoothed afterwards. An earlier version of ours handed any spell shorter than 2 s that
sat between two spells of the other team back to that team. That was our reading of "controlled
possession", and it was wrong. The public window, which the challenge names as the reference,
contains spells of 0.8 s and 1.3 s of exactly that kind, so we removed the rule. On the ten-match
check, training on unsmoothed labels was worth 0.04 `spellF1` when scored against unsmoothed
labels and cost 0.003 when scored against smoothed ones.

## Possession rules we implemented

We did not write possession rules. The models learn them from the training labels, so the rules
are those of the labels.

A change of possession happens where the training labels put it. In the Metrica games that is the
first event naming the other team as owner, and in the Bundesliga matches it is where the
official possession flag changes. A deflection, or a challenge the team in possession wins, is
not such an event. Brief possessions count, down to under a second. The decoder's penalty for
changing team stops a single odd frame from becoming one.

Loose balls and passes in flight stay with the team that last had the ball until the other team
takes it.

Dead ball runs from the whistle, or from the ball leaving the pitch, until the restart is taken.
That holds even when the ball is visible and being carried back or placed. The time before each
half's kickoff is dead, and we find the kickoff from the Smart Tagging "Kick Off (Start)" tag.

After a restart the ball belongs to whichever team the models see with it first.

## Limitations

Short possessions are the weak spot. We find most possessions longer than 5 s and fewer than half
of those under 5 s. A short spell between two long ones leaves little evidence, and missing it
also breaks the match of the two spells either side.

Short stoppages are next. On the training matches we find 63% of stoppages under 5 s, 86% of
those between 5 and 10 s, and nearly all longer ones.

When the ball is hidden or disputed, ownership gets worse. On the bench it is right on about 94%
of frames with the ball seen and 87% without. Team shape helps, but it is no substitute for the
ball.

We do not handle a ball in the air. The homography puts an airborne ball at the wrong place on
the ground, and all we do is flag those frames.

The training matches are not our match. Their labelling conventions may differ from the ones
behind this challenge's ground truth, and our damage to their tracking is a guess at our
tracker's errors.

With more time we would try a neural sequence model in place of the second classifier, and
rebuild the ball track in 3D from the two videos.

## Reproduction

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
# unpack the match data into data/tactical/ and data/broadcast/ (see DATA.md)
.venv/bin/python -m facet.run
```

This writes `evaluation/results/facet.csv`, validates it and scores it on the public window. The
first run downloads the training matches (about 2.9 GB) into `data/external/`. A run takes about
half an hour, most of it training.

Nothing is random at run time. The noise that damages the training matches has fixed seeds, the
classifiers have a fixed seed, and every version is pinned in `requirements.lock`. We cloned the
repository fresh, ran it against the data, and got a byte-identical file.

`python -m pytest tests -q` runs the tests.
