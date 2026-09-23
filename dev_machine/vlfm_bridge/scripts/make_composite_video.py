"""Builds a composite review video for a run_vlfm_pipeline.py run directory.

Layout (one frame per tick, time-aligned by tick index):
  ┌──────────────────────────────────────────┐
  │         first-person camera (960×540)    │
  ├────────────┬────────────┬────────────────┤
  │ occupancy  │  semantic  │  scene graph   │
  │   map      │    map     │  (objects)     │
  │ (320×320)  │ (320×320)  │  (320×320)     │
  └────────────┴────────────┴────────────────┘

All three map panels share the same square crop window (union of every
non-background pixel across the whole run, padded) so they stay
co-registered and don't jitter. Semantic and scene-graph panels fall back
to occupancy-map content for ticks where they have no saved image.

Usage:
    python make_composite_video.py /path/to/vlfm_pipeline_XXXX \\
        [--out first_person.mp4] [--fps 1] [--margin 40] [--keep-orig]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess

import cv2
import numpy as np

PANEL_W = 960           # total video width
FP_H = 540              # first-person panel height (16:9 at PANEL_W)
MAP_SIDE = PANEL_W // 3  # 320 px per map tile (three side by side)
BG = (24, 24, 24)


def tick_of(path):
    return int(re.search(r"tick_(\d+)", os.path.basename(path)).group(1))


def load_by_tick(folder, ext):
    return {tick_of(p): p for p in glob.glob(os.path.join(folder, f"tick_*.{ext}"))
            if not p.endswith("_vis.jpg")}


def union_crop(paths, margin):
    """Square crop window covering every non-background pixel across all maps."""
    x0 = y0 = 10**9
    x1 = y1 = -1
    h = w = None
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        # occupancy maps: white background; semantic maps: dark background
        # Use "any pixel deviates from corners" as non-background heuristic
        bg = im[0, 0]
        nz = np.argwhere(np.any(im != bg, axis=2))
        if len(nz) == 0:
            continue
        (ya, xa), (yb, xb) = nz.min(0), nz.max(0)
        x0, y0, x1, y1 = min(x0, xa), min(y0, ya), max(x1, xb), max(y1, yb)
    if h is None:
        return 0, 0, 100, 100
    if x1 < 0:
        return 0, 0, w, h
    side = int(max(x1 - x0, y1 - y0) + 1 + 2 * margin)
    side = min(side, h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    xs = int(np.clip(cx - side // 2, 0, w - side))
    ys = int(np.clip(cy - side // 2, 0, h - side))
    return xs, ys, side, side


def label(img, text, color=(255, 255, 255), org=(6, 18), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="first_person.mp4")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="ticks per second of video time (1 = real-time tick rate)")
    ap.add_argument("--margin", type=int, default=40,
                    help="map crop padding in map pixels")
    ap.add_argument("--keep-orig", action="store_true", default=True,
                    help="rename existing --out to <name>_orig.mp4 before writing")
    a = ap.parse_args()

    frames = load_by_tick(os.path.join(a.run_dir, "frames"), "jpg")
    occ    = load_by_tick(os.path.join(a.run_dir, "occupancy_map"), "png")
    sem    = load_by_tick(os.path.join(a.run_dir, "semantic_map"),  "png")
    sgr    = load_by_tick(os.path.join(a.run_dir, "scene_graph"),   "png")

    ticks = sorted(frames)
    assert ticks and occ, "need frames/ and occupancy_map/ in the run dir"

    # Crop window: derive from all three map types combined
    all_map_paths = list(occ.values()) + list(sem.values()) + list(sgr.values())
    xs, ys, cw, ch = union_crop(all_map_paths or list(occ.values()), a.margin)
    print(f"{len(ticks)} ticks, crop {xs},{ys} size {cw}x{ch} → {MAP_SIDE}x{MAP_SIDE}")

    last_occ_tick = max(occ) if occ else ticks[-1]
    frozen_after = last_occ_tick if last_occ_tick < ticks[-1] else None

    out_path = os.path.join(a.run_dir, a.out)
    if os.path.exists(out_path) and a.keep_orig:
        orig = out_path[:-4] + "_orig.mp4"
        if not os.path.exists(orig):
            shutil.move(out_path, orig)
            print("original kept as", orig)

    H = FP_H + MAP_SIDE
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{PANEL_W}x{H}", "-framerate", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", "30", out_path],
        stdin=subprocess.PIPE)

    def latest_at_or_before(d, t, fallback=None):
        valid = [x for x in d if x <= t]
        if not valid:
            return None, fallback
        k = max(valid)
        return k, d[k]

    def map_tile(path, title, held_from=None):
        im = cv2.imread(path)
        if im is None:
            im = np.full((ch, cw, 3), BG, dtype=np.uint8)
        else:
            im = im[ys:ys + ch, xs:xs + cw]
        im = cv2.resize(im, (MAP_SIDE, MAP_SIDE), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(im, (0, 0), (MAP_SIDE - 1, MAP_SIDE - 1), (100, 100, 100), 1)
        label(im, title)
        if held_from is not None:
            label(im, f"held tick {held_from}", (0, 200, 255),
                  (6, MAP_SIDE - 8), 0.4)
        return im

    for t in ticks:
        # --- first-person panel ---
        fp = cv2.resize(cv2.imread(frames[t]), (PANEL_W, FP_H), interpolation=cv2.INTER_CUBIC)
        label(fp, f"tick {t}/{ticks[-1]}", (255, 255, 255), (10, FP_H - 18), 0.65)
        bar_y = FP_H - 6
        cv2.rectangle(fp, (0, bar_y), (PANEL_W, FP_H), (60, 60, 60), -1)
        cv2.rectangle(fp, (0, bar_y),
                      (int(PANEL_W * (t - ticks[0] + 1) / len(ticks)), FP_H),
                      (0, 200, 0), -1)
        if frozen_after:
            fx = int(PANEL_W * (frozen_after - ticks[0] + 1) / len(ticks))
            cv2.line(fp, (fx, bar_y - 6), (fx, FP_H), (0, 200, 255), 2)

        # --- three map panels ---
        ko, po = latest_at_or_before(occ, t)
        ks, ps = latest_at_or_before(sem, t)
        kg, pg = latest_at_or_before(sgr, t)

        # Semantic / scene-graph fall back to occupancy if not yet computed
        tile_occ = map_tile(po, "OCCUPANCY",  ko if ko != t else None)
        tile_sem = map_tile(ps or po, "SEMANTIC MAP",   ks if (ks and ks != t) else None)
        tile_sgr = map_tile(pg or po, "SCENE GRAPH",   kg if (kg and kg != t) else None)

        canvas = np.full((H, PANEL_W, 3), BG, np.uint8)
        canvas[:FP_H] = fp
        canvas[FP_H:, :MAP_SIDE]            = tile_occ
        canvas[FP_H:, MAP_SIDE:2*MAP_SIDE]  = tile_sem
        canvas[FP_H:, 2*MAP_SIDE:]          = tile_sgr
        ff.stdin.write(canvas.tobytes())

    ff.stdin.close()
    ff.wait()
    print("wrote", out_path, f"({PANEL_W}x{H}, {len(ticks)} ticks @ {a.fps} tick/s)")


if __name__ == "__main__":
    main()
