"""Look at a stretch of the match: a pitch clip, the tracking drawn over the video, and a timeline.

    .venv/bin/python tools/viewer.py --start 11185 --seconds 20
    .venv/bin/python tools/viewer.py --start 11185 --labels evaluation/results/baseline_smooth50.csv

Writes to data/viz/, which is gitignored. Everything stays on this machine: the match data must
not be redistributed, so never commit or publish these files.

The tracking is drawn over the video through the homography, which assumes everything is on the
ground. A ball in the air is therefore drawn in the wrong place (further from the camera than the
real ball). That mismatch is the thing to watch for.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, GAME, require  # noqa: E402
from starter import load_homography  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
from starter.utils import AWAY_COLOUR, BALL_COLOUR, HOME_COLOUR, draw_pitch  # noqa: E402

FPS = 25
OUT_DIR = os.path.join(ROOT, "data", "viz")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
LABEL_COLOURS = {"H": HOME_COLOUR, "A": AWAY_COLOUR, "D": "#7f7f7f", "X": "#ffffff"}
OVERLAY_SCALE = 0.5  # video drawn at 960x540 to keep the clip small


def read_tracking(raw_path, start, n):
    """(players [n,46,3], ball [n,3]) for video frames start..start+n-1, normalised 0-1 coords."""
    players, ball = [], []
    with open(raw_path) as handle:
        for line in handle:
            frame_id, slots, b = line.split(":")
            video_frame = int(frame_id) - 1
            if video_frame < start:
                continue
            if video_frame >= start + n:
                break
            players.append([[float(v) for v in g.split(",")] for g in slots.split(";")])
            ball.append([float(v) for v in b.split(",")])
    assert len(ball) == n, "range runs past the end of the tracking"
    return np.array(players), np.array(ball)


def read_labels(path, start, n):
    return pd.read_csv(path)["label"].to_numpy()[start:start + n].astype(str)


def timeline(path, frames, players, ball, label_rows, pitch):
    import matplotlib.pyplot as plt

    seconds = (frames - frames[0]) / FPS
    xy = ball[:, :2] * pitch
    ball_speed = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) * FPS]
    player_speed = np.nanmean(players[:, :, 2], axis=1)

    rows = 2 + len(label_rows)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 1.6 * rows), sharex=True)
    axes[0].plot(seconds, ball_speed, lw=0.8, color="black", label="ball speed (from positions)")
    axes[0].plot(seconds, player_speed, lw=1.2, color="tab:orange", label="mean player speed")
    axes[0].set_ylabel("m/s")
    axes[0].set_ylim(0, 40)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].fill_between(seconds, 0, np.isnan(ball[:, 0]), step="mid", color="tab:red")
    axes[1].set_ylabel("ball\nmissing")
    axes[1].set_yticks([])
    for ax, (name, labels) in zip(axes[2:], label_rows):
        ax.scatter(seconds, np.zeros_like(seconds), c=[LABEL_COLOURS[l] for l in labels],
                   marker="|", s=400)
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)
        ax.set_yticks([])
    axes[-1].set_xlabel("seconds from frame {}".format(frames[0]))
    fig.suptitle("H red · A blue · D grey · X white", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pitch_clip(path, frames, players, ball, team, label_rows, pitch):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    length, width = pitch
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    draw_pitch(ax, length, width)
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-5, width + 5)
    colours = np.where(team == 0, HOME_COLOUR, AWAY_COLOUR)
    dots = ax.scatter([], [], s=80, edgecolors="black", zorder=3)
    trail, = ax.plot([], [], color=BALL_COLOUR, lw=1, alpha=0.6, zorder=4)
    ball_dot = ax.scatter([], [], s=50, c=BALL_COLOUR, edgecolors="black", zorder=5)
    title = ax.set_title("")

    writer = FFMpegWriter(fps=FPS)
    with writer.saving(fig, path, dpi=80):
        for i, frame in enumerate(frames):
            seen = ~np.isnan(players[i, :, 0])
            dots.set_offsets(players[i, seen, :2] * pitch)
            dots.set_facecolors(colours[seen])
            recent = ball[max(0, i - FPS):i + 1, :2] * pitch
            trail.set_data(recent[:, 0], recent[:, 1])
            ball_dot.set_offsets(ball[i:i + 1, :2] * pitch if not np.isnan(ball[i, 0])
                                 else np.empty((0, 2)))
            tags = "   ".join("{} {}".format(name, labels[i]) for name, labels in label_rows)
            title.set_text("frame {}   t={:.1f}s   {}".format(frame, i / FPS, tags))
            writer.grab_frame()
    plt.close(fig)


def overlay_clip(path, frames, players, ball, team, video_path):
    """The tactical video with the tracking projected onto it through the homography."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    matrices, _, valid = load_homography.load()
    w, h = int(1920 * OVERLAY_SCALE), int(1080 * OVERLAY_SCALE)
    decode = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(frames[0] / FPS), "-i", video_path,
         "-frames:v", str(len(frames)), "-vf", "scale={}:{}".format(w, h),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)

    fig = plt.figure(figsize=(w / 80, h / 80), dpi=80)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    image = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
    colours = np.where(team == 0, HOME_COLOUR, AWAY_COLOUR)
    dots = ax.scatter([], [], s=60, facecolors="none", linewidths=2)
    ball_dot = ax.scatter([], [], s=90, facecolors="none", edgecolors="yellow", linewidths=2)

    writer = FFMpegWriter(fps=FPS)
    with writer.saving(fig, path, dpi=80):
        for i, frame in enumerate(frames):
            raw = decode.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            image.set_data(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
            seen = ~np.isnan(players[i, :, 0])
            if valid[frame] and seen.any():
                px, py = load_homography.project(matrices[frame], players[i, seen, 0],
                                                 players[i, seen, 1])
                dots.set_offsets(np.c_[px, py] * OVERLAY_SCALE)
                dots.set_edgecolors(colours[seen])
            else:
                dots.set_offsets(np.empty((0, 2)))
            if valid[frame] and not np.isnan(ball[i, 0]):
                bx, by = load_homography.project(matrices[frame], ball[i, 0], ball[i, 1])
                ball_dot.set_offsets([[bx * OVERLAY_SCALE, by * OVERLAY_SCALE]])
            else:
                ball_dot.set_offsets(np.empty((0, 2)))
            writer.grab_frame()
    decode.wait()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, required=True, help="first video frame")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--labels", nargs="*", default=[], help="submission CSVs to show")
    parser.add_argument("--no-video", action="store_true", help="skip the video overlay clip")
    args = parser.parse_args()

    meta_path, raw_path, video_path = require(["atd_meta", "atd_raw", "video"], GAME)
    n = int(args.seconds * FPS)
    frames = np.arange(args.start, args.start + n)
    pitch = np.array(GAMES[GAME]["pitch"])
    team = team_of_each_track(meta_path)
    players, ball = read_tracking(raw_path, args.start, n)

    label_rows = [(os.path.splitext(os.path.basename(p))[0], read_labels(p, args.start, n))
                  for p in args.labels]
    public = read_labels(PUBLIC_GT, args.start, n)
    if (public != "X").any():
        label_rows.append(("public answer", public))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, "{}_{}s".format(args.start, int(args.seconds)))
    timeline(stem + "_timeline.png", frames, players, ball, label_rows, pitch)
    print("wrote", stem + "_timeline.png")
    pitch_clip(stem + "_pitch.mp4", frames, players, ball, team, label_rows, pitch)
    print("wrote", stem + "_pitch.mp4")
    if not args.no_video:
        overlay_clip(stem + "_video.mp4", frames, players, ball, team, video_path)
        print("wrote", stem + "_video.mp4")


if __name__ == "__main__":
    main()
