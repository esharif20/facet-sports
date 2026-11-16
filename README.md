# Facet Sports

This repository is our team's entry for More than a Hack 2027, built on the official
challenge repository (kept as the `upstream` remote). Our code lives in `facet/`, and
[docs/technical_document.md](docs/technical_document.md) explains the method. The upstream files
are left unchanged so organiser fixes merge cleanly with `git pull upstream main`.

## Team quickstart

1. **Add your copy of the organiser-provided match data.** The licensed match data is excluded
   from this public repository. Use the same FC Barcelona vs Athletic Club bundle supplied for
   the challenge; no separate upload or import command is needed. From the repository root,
   unpack or copy the files into this layout, keeping their original names:

   ```text
   data/
   ├── tactical/
   │   ├── BAR-ATH_tactical.mp4
   │   ├── BAR-ATH_tactical_atd.xml
   │   ├── BAR-ATH_tactical_atd.txt
   │   ├── BAR-ATH_tactical_homography.csv
   │   └── BAR-ATH_tactical_smart_tagging.xml
   └── broadcast/
       ├── BAR-ATH_broadcast.mp4
       ├── BAR-ATH_broadcast_atd.xml
       ├── BAR-ATH_broadcast_atd.txt
       └── BAR-ATH_broadcast_homography.csv
   ```

   If you already keep the bundle elsewhere, set `export MTAH_DATA_DIR="/absolute/path/to/bundle"`
   in the terminal before running the commands below. Point it at the parent directory containing
   both `tactical/` and `broadcast/`, not at either recording folder. The pipeline reads the files
   automatically. See [DATA.md](DATA.md) for asset details. Everything in `data/` is gitignored;
   do not commit or redistribute the match data.

2. **Environment** (Python 3.11, pinned versions in `requirements.lock`):

   ```bash
   uv venv --python 3.11 .venv
   uv pip install --python .venv/bin/python -r requirements.lock
   ```

3. **Check that everything runs:**

   ```bash
   .venv/bin/python evaluation/run_evaluation.py   # scores the four baselines, no data needed
   .venv/bin/python starter/baseline.py            # needs the data; rebuilds the baselines
   ```

4. **Look at the data** (writes clips to `data/viz/`, which is gitignored; never publish them):

   ```bash
   .venv/bin/python tools/viewer.py --start 11185 --seconds 20 --labels evaluation/results/baseline_smooth50.csv
   ```

   You get a timeline (ball speed, player speed, when the ball is missing, labels), a pitch clip, and
   the tracking drawn over the tactical video. The homography assumes everything is on the ground,
   so a ball in the air is placed further from the camera than it really is.

5. **Build our submission** (about half an hour; writes `evaluation/results/facet.csv`, validates it
   and scores it on the public 5 minutes as a sanity check):

   ```bash
   .venv/bin/python -m facet.run
   .venv/bin/python -m pytest tests -q
   ```

   The first run downloads the training matches into `data/external/`: Metrica's three open sample
   games (about 240 MB) and seven open Bundesliga matches (about 2.6 GB).

6. **Score a change before trusting it** (about an hour):

   ```bash
   .venv/bin/python -m facet.bench          # the current settings
   .venv/bin/python -m facet.bench sweep    # a grid around them
   ```

   The bench runs the pipeline on Metrica's three sample games with the official metric, each scored
   by models that never saw it. Settings are chosen there, never on the competition ground truth.

## How it works

| Step | File | What it does |
|---|---|---|
| Load | `facet/load.py` | Both feeds' tracking as arrays, cached in `data/cache/`. |
| Sync | `facet/sync.py` | Finds the broadcast-to-tactical frame offset of each half by sliding one ball track along the other. |
| Fuse | `facet/ball/fuse.py` | One ball track: tactical, with broadcast filling gaps. Frames where the two feeds disagree by more than 3 m are marked disputed (ball in the air, or one tracker wrong). |
| Ghosts | `facet/players.py` | Removes a player tracked twice. |
| Other matches | `facet/sample_games.py` | Ten matches with known possession: Metrica's three open sample games and seven open Bundesliga matches, roughened until their ball looks like our tracking. |
| Dead ball | `facet/dead.py` | A classifier of how players and ball move during a stoppage, trained on the other matches. |
| Ownership | `facet/owner.py` | A classifier of which team owns the ball, from the contacts and touches each team has with it and from how the two teams are arranged, trained on the other matches. |
| Second look | `facet/refine.py` | A third classifier that reads the first two's answers over the surrounding seconds and corrects ownership. Trained on answers given for matches the first two had not seen. |
| Decode | `facet/decode.py` | A three-state Viterbi (home, away, dead) that turns the two probabilities into whole possessions. |
| Bench | `facet/bench.py` | Scores all of the above on Metrica's sample games. |

**Rules we follow** (see [RULES.md](RULES.md)):
- Every label comes out of code. There is no hand labelling or hand correction of this game.
- The public 5-minute ground truth is only for checking that the pipeline runs and that our labels
  mean what the challenge's labels mean. We do not train or tune on it.
- Before every push, run `git ls-files | grep -E "_atd\.(txt|xml)|_homography\.csv|_smart_tagging\.xml|\.mp4|\.npz"`.
  It must print nothing.

---

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
