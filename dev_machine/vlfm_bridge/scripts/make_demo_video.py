"""Builds the final demo video for a run_vlfm_pipeline.py run: the externally
filmed third-person video (real time, 30 fps) is the main picture and the
pipeline's own 1 Hz artifacts are inserted at the moment they actually
happened:

    +-----------------------------------+--------------------------+
    |                                   |                          |
    |        THIRD-PERSON VIEW          |   FIRST-PERSON (robot    |
    |        (main, 16:9)               |   camera + detections)   |
    |                                   |        (4:3)             |
    +----------------+------------------+--------------+-----------+
    | OCCUPANCY MAP  |  status / phase  |  VALUE MAP   | legend    |
    +----------------+------------------+--------------+-----------+

HOW THE TIME ALIGNMENT WORKS (the important part)
  * Every tick's artifacts carry the wall-clock time they were written:
    frames/tick_N.jpg = the camera image + detections of tick N,
    occupancy_map/tick_N.png and value_map/tick_N.png = the maps after tick N.
    Those file mtimes are the only high-precision clock the pipeline leaves
    behind, so the video is aligned against them, not against the log's
    relative "t=" values.
  * The third-person camera was started by hand, so one sync point is needed:
    --first-tick-at S = the video second at which tick 1 happened. It is found
    once per recording by matching the robot's stop-and-go pattern in the video
    (8 in-place rotations of the initial scan, each followed by a ~2 s pause)
    against the rotation windows in cmdvel_udp_relay.log + log.txt; the pattern
    is unique, so the match is good to a few tenths of a second.
  * A first-person frame is shown from its capture time (file mtime minus
    --fp-lead, the detector latency) until the next tick's frame; maps are
    shown from the moment they were saved until the next map. After the target
    lock the pipeline stops updating the maps, so the last map is held and
    labelled as such.

Usage:
    python make_demo_video.py RUN_DIR [--third RUN_DIR/ThirdPersonView.mp4]
        [--first-tick-at 6.05] [--out demo.mp4] [--lead-in 3.0]
"""
import argparse
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- layout (px)
W_LEFT, W_RIGHT = 1120, 840            # third-person 16:9 / first-person 4:3, same height
H_TOP = 630
H_BOT = 400
CANVAS_W, CANVAS_H = W_LEFT + W_RIGHT, H_TOP + H_BOT
MAP = 380                              # each map tile (square), 10 px padding inside a 400 cell
BG = (22, 22, 22)
PANEL = (34, 34, 34)
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"

PHASES = [  # key, label, RGB
    ("standby", "STANDBY", (150, 150, 150)),
    ("scan", "1  INITIAL 360° SCAN", (90, 200, 255)),
    ("explore", "2  EXPLORING", (255, 255, 255)),
    ("walk", "3  TARGET LOCKED → WALKING TO GOAL", (255, 170, 60)),
    ("found", "4  FOUND — STOPPED", (80, 230, 110)),
]
PHASE_RGB = {k: c for k, _, c in PHASES}
PHASE_LABEL = {k: l for k, l, _ in PHASES}


def font(size, bold=False):
    return ImageFont.truetype(FONT_DIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


F_S, F_M, F_L, F_XL = font(15), font(19), font(25, True), font(30, True)


# ------------------------------------------------------------------ log parse
def parse_log(run_dir):
    """tick -> dict(kind, text fields) from log.txt."""
    info, rot_dur = {}, []
    scan_re = re.compile(r"\[scan (\d+)/8[^\]]*tick (\d+)\] pose=\(([-+\d.]+),([-+\d.]+)\) yaw=([-+\d]+)deg")
    main_re = re.compile(r"\[tick\s+(\d+) t=\s*([\d.]+)s target='([^']*)'\] pose=\(([-+\d.]+),([-+\d.]+)\)(.*)")
    rot_re = re.compile(r"\[scan (\d+)/8\] rotate to ([-+\d]+)deg: \w+ \(.*?, ([\d.]+)s\)")
    found = None
    target = "?"
    for line in open(os.path.join(run_dir, "log.txt"), errors="replace"):
        m = scan_re.search(line)
        if m:
            info[int(m[2])] = dict(kind="scan", step=int(m[1]), pose=(float(m[3]), float(m[4])), yaw=int(m[5]))
            continue
        m = rot_re.search(line)
        if m:
            rot_dur.append((int(m[1]), int(m[2]), float(m[3])))
            continue
        m = main_re.search(line)
        if m:
            rest, d = m[6], dict(kind="main", pose=(float(m[4]), float(m[5])))
            target = m[3]
            if (w := re.search(r"Walking toward the goal \(([^)]*)\) dist=([\d.]+)m", rest)):
                d.update(kind="walk", goal=w[1], dist=float(w[2]))
            if (k := re.search(r"LOCKED goal=\(([^)]*)\) dist=([\d.]+)m", rest)):
                d.update(kind="lock", goal=k[1], dist=float(k[2]))
            if (s := re.search(r"score=([-+\d.]+)", rest)):
                d["score"] = float(s[1])
            if (s := re.search(r"saw '([^']*)' conf=([\d.]+) hits=(\d)/", rest)):
                d["saw"] = (s[1], float(s[2]), int(s[3]))
            info[int(m[1])] = d
            continue
        m = re.search(r"FOUND '([^']*)' at \(([^)]*)\), ([\d.]+)m away", line)
        if m:
            found = dict(target=m[1], at=m[2], dist=float(m[3]))
    return info, rot_dur, found, target


def relay_stop_times(run_dir):
    """Wall-clock times at which cmdvel_udp_relay went silent = end of each scan rotation."""
    p = os.path.join(run_dir, "cmdvel_udp_relay.log")
    if not os.path.exists(p):
        return []
    return [float(m[1]) for m in re.finditer(r"\[INFO\] \[([\d.]+)\] \[cmdvel_udp_relay\]: command stream stopped",
                                              open(p, errors="replace").read())]


def by_tick(folder, ext):
    out = {}
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            m = re.fullmatch(r"tick_(\d+)\." + ext, f)
            if m:
                out[int(m[1])] = os.path.join(folder, f)
    return out


def mtime(p):
    return os.stat(p).st_mtime


# ------------------------------------------------------------------ map tiles
def union_crop(paths, margin=30):
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    h = w = 800
    for p in paths:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        nz = np.argwhere(np.any(im != 255, axis=2))
        if len(nz):
            (ya, xa), (yb, xb) = nz.min(0), nz.max(0)
            x0, y0, x1, y1 = min(x0, xa), min(y0, ya), max(x1, xb), max(y1, yb)
    if x1 < 0:
        return 0, 0, w
    side = min(int(max(x1 - x0, y1 - y0) + 1 + 2 * margin), h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return int(np.clip(cx - side // 2, 0, w - side)), int(np.clip(cy - side // 2, 0, h - side)), side


def text_cv(img, txt, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def map_tile(path, crop, title, held_from=None):
    if path is None:
        im = np.full((MAP, MAP, 3), 255, np.uint8)
        text_cv(im, "waiting for first update", (60, MAP // 2), 0.6, (120, 120, 120))
    else:
        x, y, s = crop
        im = cv2.resize(cv2.imread(path)[y:y + s, x:x + s], (MAP, MAP), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (MAP - 1, MAP - 1), (150, 150, 150), 1)
    cv2.rectangle(im, (0, 0), (215, 34), (30, 30, 30), -1)
    cv2.putText(im, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if held_from is not None:
        cv2.rectangle(im, (0, MAP - 30), (MAP - 1, MAP - 1), (30, 30, 30), -1)
        cv2.putText(im, f"frozen at tick {held_from}: target locked", (10, MAP - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 165, 255), 1, cv2.LINE_AA)
    return im


def bar_text(img, txt):
    """Caption strip along the bottom edge of a video panel."""
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    cv2.putText(img, txt, (12, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--third", default=None, help="third-person video (default RUN_DIR/ThirdPersonView.mp4)")
    ap.add_argument("--first-tick-at", type=float, required=True,
                    help="video second at which pipeline tick 1 happened (the sync point, see module docstring)")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--lead-in", type=float, default=3.0,
                    help="seconds of third-person video kept before tick 1 (default 3; the rest is trimmed)")
    ap.add_argument("--fp-lead", type=float, default=0.3,
                    help="first-person image is shown this many seconds before its file was written "
                         "(= detector latency), default 0.3")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    third = a.third or os.path.join(a.run_dir, "ThirdPersonView.mp4")
    frames = by_tick(os.path.join(a.run_dir, "frames"), "jpg")
    occ = by_tick(os.path.join(a.run_dir, "occupancy_map"), "png")
    val = by_tick(os.path.join(a.run_dir, "value_map"), "png")
    info, rot_dur, found, target = parse_log(a.run_dir)
    ticks = sorted(frames)
    t1 = mtime(frames[ticks[0]])                       # wall clock of tick 1
    to_video = lambda wall: a.first_tick_at + (wall - t1)   # wall clock -> video seconds

    fp_start = {k: to_video(mtime(frames[k])) - a.fp_lead for k in ticks}
    occ_start = {k: to_video(mtime(occ[k])) for k in occ}
    val_start = {k: to_video(mtime(val[k])) for k in val}
    last_map = max(occ)
    frozen = last_map < ticks[-1]

    first_main = min((k for k, v in info.items() if v["kind"] != "scan"), default=None)
    lock_tick = next((k for k, v in sorted(info.items()) if v["kind"] == "lock"), None)
    t_scan = fp_start[ticks[0]]
    t_main = fp_start[first_main] if first_main else 1e9
    t_lock = fp_start[lock_tick] if lock_tick else 1e9
    t_end = fp_start[ticks[-1]] + a.fp_lead + (0.3 if found else 0)     # FOUND is logged during the last tick
    # rotation windows (video seconds): end = relay's "stream stopped" minus its 0.4 s activity timeout + burst
    stops = relay_stop_times(a.run_dir)
    rot_win = []
    for (step, deg, dur), stop in zip(rot_dur, stops):
        e = to_video(stop - 0.7)
        rot_win.append((e - dur, e, step))

    crop = union_crop(list(occ.values()) + list(val.values()))
    print(f"ticks {ticks[0]}..{ticks[-1]}  maps up to tick {last_map}  map crop {crop}  "
          f"phases: scan {t_scan:.1f}s main {t_main:.1f}s lock {t_lock:.1f}s end {t_end:.1f}s  rotations {len(rot_win)}")

    cap = cv2.VideoCapture(third)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v0 = max(0.0, t_scan - a.lead_in)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(v0 * fps)))
    v_end = n_frames / fps
    span = v_end - v0
    print(f"third-person {n_frames} frames @ {fps:g}fps; using {v0:.2f}s..{v_end:.2f}s ({span:.1f}s)")

    # ------- static background: titles + legends (PIL, drawn once)
    bg = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(bg)
    d.rectangle([0, H_TOP, CANVAS_W, CANVAS_H], fill=PANEL)
    vx = W_LEFT + 400                           # legend panel x (right of the value map)
    d.line([(W_LEFT, H_TOP), (W_LEFT, CANVAS_H)], fill=(70, 70, 70), width=2)
    d.line([(0, H_TOP), (CANVAS_W, H_TOP)], fill=(70, 70, 70), width=2)
    # occupancy legend (left tile, under the status text) drawn in the dynamic part; value legend here:
    d.text((vx + 18, H_TOP + 18), "VALUE MAP", font=F_L, fill=(255, 255, 255))
    d.text((vx + 18, H_TOP + 54), "how well each area matches", font=F_S, fill=(190, 190, 190))
    d.text((vx + 18, H_TOP + 74), "the target text (BLIP-2 ITM),", font=F_S, fill=(190, 190, 190))
    d.text((vx + 18, H_TOP + 94), "fused over camera views", font=F_S, fill=(190, 190, 190))
    cb_x, cb_y, cb_w, cb_h = vx + 18, H_TOP + 140, 34, 200
    grad = np.linspace(255, 0, cb_h).astype(np.uint8)[:, None].repeat(cb_w, 1)
    cbar = cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)[..., ::-1]
    bg.paste(Image.fromarray(np.ascontiguousarray(cbar)), (cb_x, cb_y))
    d.rectangle([cb_x, cb_y, cb_x + cb_w, cb_y + cb_h], outline=(200, 200, 200))
    d.text((cb_x + cb_w + 12, cb_y - 4), "high  → look here", font=F_S, fill=(255, 220, 120))
    d.text((cb_x + cb_w + 12, cb_y + cb_h - 18), "low", font=F_S, fill=(170, 170, 170))
    d.text((cb_x + cb_w + 12, cb_y + cb_h // 2 - 10), "medium", font=F_S, fill=(170, 170, 170))
    d.text((vx + 18, H_TOP + 350), "white = not yet observed", font=F_S, fill=(190, 190, 190))
    d.text((vx + 18, H_TOP + 372), "green line = robot path", font=F_S, fill=(0, 230, 0))
    bg_np = np.array(bg)[..., ::-1].copy()

    # ------- caches
    fp_cache, occ_cache, val_cache = {}, {}, {}

    def fp_tile(k):
        if k not in fp_cache:
            if k is None:
                im = np.full((H_TOP, W_RIGHT, 3), 28, np.uint8)
                text_cv(im, "waiting for pipeline start", (230, H_TOP // 2), 0.8, (150, 150, 150), 1)
            else:
                im = cv2.resize(cv2.imread(frames[k]), (W_RIGHT, H_TOP), interpolation=cv2.INTER_CUBIC)
                bar_text(im, f"FIRST-PERSON  robot RGB camera + detector    tick {k}   (snapshot ~1 Hz)")
            fp_cache[k] = im
        return fp_cache[k]

    def map_pick(starts, files, cache, t, title, side_frozen):
        ks = [k for k, s in starts.items() if s <= t]
        k = max(ks) if ks else None
        held = k if (frozen and k == last_map and side_frozen) else None
        if (k, held) not in cache:
            cache[(k, held)] = map_tile(files[k] if k is not None else None, crop, title, held)
        return cache[(k, held)]

    def phase_at(t):
        if t < t_scan:
            return "standby"
        if t < t_main:
            return "scan"
        if t < t_lock:
            return "explore"
        if t < t_end:
            return "walk"
        return "found"

    def status_panel(t, tick):
        ph = phase_at(t)
        im = Image.new("RGB", (W_LEFT - 400, H_BOT), PANEL)
        dd = ImageDraw.Draw(im)
        dd.text((18, 14), PHASE_LABEL[ph], font=F_L, fill=PHASE_RGB[ph])
        lines = []
        kinfo = info.get(tick, {}) if tick else {}
        if ph == "scan":
            win = next((w for w in rot_win if w[0] <= t <= w[1]), None)
            step = win[2] if win else kinfo.get("step", 0)
            lines.append(f"rotating in place to the next 45° heading (closed loop on SLAM yaw)"
                         if win else "holding still: BLIP-2 value map + YOLO-World detection")
            lines.append(f"scan step {step}/8   •   target: '{target}'")
        elif ph in ("explore", "standby"):
            lines.append("VLFM: value map → best frontier → waypoint → local planner avoids obstacles")
            lines.append(f"target: '{target}'" + (f"   value score {kinfo['score']:+.2f}" if "score" in kinfo else ""))
        elif ph == "walk":
            lines.append("target locked (median of 3 detections); upper layer stops reasoning")
            lines.append("local planner + path follower drive to the goal")
        else:
            lines.append(f"reached '{found['target']}' at ({found['at']}), {found['dist']:.2f} m away → stop, lie down"
                         if found else "done")
        y = 66
        max_w = W_LEFT - 400 - 36
        for ln in lines:
            words, cur = ln.split(" "), ""
            for wd in words:                       # wrap to the panel width
                if dd.textlength((cur + " " + wd).strip(), font=F_M) > max_w:
                    dd.text((18, y), cur, font=F_M, fill=(225, 225, 225))
                    y += 28
                    cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            dd.text((18, y), cur, font=F_M, fill=(225, 225, 225))
            y += 30
        # detection line
        saw = kinfo.get("saw")
        if saw and ph in ("explore", "scan"):
            dd.text((18, y), f"detected '{saw[0]}'  conf {saw[1]:.2f}   (hit {saw[2]} of 3 needed to lock)", font=F_M,
                    fill=(255, 230, 90))
        elif ph == "walk" and "dist" in kinfo:
            dd.text((18, y), f"goal ({kinfo['goal']})    distance {kinfo['dist']:.2f} m", font=F_M, fill=(255, 190, 90))
        elif ph == "found":
            dd.text((18, y), "arrival threshold 0.25 m reached", font=F_M, fill=(110, 240, 130))
        y += 36
        # occupancy legend
        dd.text((18, y), "OCCUPANCY MAP", font=F_S, fill=(190, 190, 190))
        chips = [((200, 255, 200), "explored free"), ((100, 100, 100), "obstacle (inflated)"),
                 ((0, 0, 0), "obstacle"), ((0, 0, 200), "frontier"), ((0, 255, 0), "path"), ((15, 192, 255), "robot")]
        x = 18
        for col, name in chips:
            dd.rectangle([x, y + 26, x + 18, y + 44], fill=col, outline=(200, 200, 200))
            dd.text((x + 26, y + 26), name, font=F_S, fill=(210, 210, 210))
            x += 26 + int(dd.textlength(name, font=F_S)) + 22
        # timeline
        tl_y, tl_x0, tl_x1 = H_BOT - 62, 18, W_LEFT - 400 - 18
        segs = [("scan", t_scan, t_main), ("explore", t_main, t_lock), ("walk", t_lock, t_end), ("found", t_end, v_end)]
        segs = [(k, a0, min(b0, v_end)) for k, a0, b0 in segs if a0 < v_end and b0 > a0]
        px = lambda tt: tl_x0 + (tl_x1 - tl_x0) * (min(max(tt, v0), v_end) - v0) / span
        dd.rectangle([tl_x0, tl_y, tl_x1, tl_y + 14], fill=(55, 55, 55))
        for k, a0, b0 in segs:
            dd.rectangle([px(a0), tl_y, px(b0), tl_y + 14], fill=tuple(int(c * 0.55) for c in PHASE_RGB[k]))
            dd.text((px(a0) + 4, tl_y + 20), {"scan": "scan", "explore": "explore", "walk": "walk", "found": ""}[k],
                    font=F_S, fill=(170, 170, 170))
        dd.rectangle([px(t) - 2, tl_y - 5, px(t) + 2, tl_y + 19], fill=(255, 255, 255))
        dd.text((tl_x1 - 92, tl_y - 30), f"{t - t_scan:+6.1f} s", font=F_S, fill=(230, 230, 230))
        return np.array(im)[..., ::-1]

    # ------- render
    tmp = os.path.join(a.run_dir, ".demo_noaudio.mp4")
    out = os.path.join(a.run_dir, a.out)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{CANVAS_W}x{CANVAS_H}", "-framerate", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", tmp],
        stdin=subprocess.PIPE)
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = v0 + idx / fps
        idx += 1
        ks = [k for k in ticks if fp_start[k] <= t]
        tick = max(ks) if ks else None
        canvas = bg_np.copy()
        tp = cv2.resize(fr, (W_LEFT, H_TOP), interpolation=cv2.INTER_AREA)
        bar_text(tp, "THIRD-PERSON VIEW")
        canvas[:H_TOP, :W_LEFT] = tp
        canvas[:H_TOP, W_LEFT:] = fp_tile(tick)
        canvas[H_TOP + 10:H_TOP + 10 + MAP, 10:10 + MAP] = map_pick(occ_start, occ, occ_cache, t, "OCCUPANCY MAP", True)
        canvas[H_TOP + 10:H_TOP + 10 + MAP, W_LEFT + 10:W_LEFT + 10 + MAP] = map_pick(
            val_start, val, val_cache, t, "VALUE MAP", True)
        canvas[H_TOP:, 400:W_LEFT] = status_panel(t, tick)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()

    if a.no_audio:
        os.replace(tmp, out)
    else:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ss", f"{v0:.3f}", "-i", third,
                            "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", out])
        if r.returncode == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, out)
            print("audio mux failed; wrote video without audio", file=sys.stderr)
    print(f"wrote {out}  ({CANVAS_W}x{CANVAS_H}, {idx} frames, {idx / fps:.1f}s)")


if __name__ == "__main__":
    main()
