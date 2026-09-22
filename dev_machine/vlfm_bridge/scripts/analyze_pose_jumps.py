#!/usr/bin/env python3
"""Offline analysis of a `ros2 bag record` made while the robot moved, to find
WHERE the pose jumps that show up on the maps are introduced.

Chain being audited (arise_slam_mid360):
    lidar -> laser_mapping_node -> /aft_mapped_to_init_incremental (raw scan-to-map pose)
    imu_preintegration_node fuses that with /livox/imu -> /state_estimation (what the
    pipeline actually reads via pose_udp_relay)
so for every step of /state_estimation bigger than --jump-m we look at (a) whether the
raw lidar-mapping pose jumped at the same time, (b) whether the reference odometry
/utlidar/robot_odom (Unitree's own leg/IMU estimate) agrees that the robot moved,
(c) laser-mapping match statistics, (d) the failureDetected/health flag.

Usage (needs ROS sourced, run with /usr/bin/python3):
    python3 analyze_pose_jumps.py /home/tommy/Taowen/RES/pose_diag_XXXX [--jump-m 0.15]
"""
import argparse
import bisect
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cls = {n: get_message(t) for n, t in types.items()}
    data = {n: [] for n in types}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        data[topic].append((t_ns * 1e-9, deserialize_message(raw, msg_cls[topic])))
    return data


def odom_arr(rows):
    """rows -> (t, x, y, z, yaw) arrays."""
    t = np.array([r[0] for r in rows])
    x = np.array([r[1].pose.pose.position.x for r in rows])
    y = np.array([r[1].pose.pose.position.y for r in rows])
    z = np.array([r[1].pose.pose.position.z for r in rows])
    yaw = np.array([yaw_of(r[1].pose.pose.orientation) for r in rows])
    return t, x, y, z, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--jump-m", type=float, default=0.15,
                    help="a step between consecutive /state_estimation samples larger than this (m) is a jump")
    a = ap.parse_args()

    d = load(a.bag)
    for n, rows in sorted(d.items()):
        if rows:
            dur = rows[-1][0] - rows[0][0]
            print(f"{n:38s} {len(rows):7d} msgs  {len(rows) / max(dur, 1e-9):7.1f} Hz")
    se = d.get("/state_estimation") or sys.exit("no /state_estimation in bag")
    t, x, y, z, yaw = odom_arr(se)
    step = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(t)
    print(f"\n/state_estimation: {len(t)} samples over {t[-1] - t[0]:.1f}s; "
          f"step median {np.median(step) * 100:.2f}cm p99 {np.percentile(step, 99) * 100:.1f}cm max {step.max():.2f}m; "
          f"max dt {dt.max() * 1000:.0f}ms")
    jumps = np.where(step > a.jump_m)[0]
    print(f"jumps > {a.jump_m}m: {len(jumps)}\n")

    def series(name):
        rows = d.get(name) or []
        return odom_arr(rows) if rows else None

    raw = series("/aft_mapped_to_init_incremental")
    lo = series("/laser_odometry")
    ref = series("/utlidar/robot_odom")
    health = d.get("/state_estimation_health") or []
    stats = d.get("/arise_slam_mid360_stats") or []
    stats_t = [r[0] for r in stats]

    def at(s, tt):
        if s is None:
            return None
        i = bisect.bisect_right(s[0], tt) - 1
        return None if i < 0 else i

    def disp(s, t0, t1):
        """xy displacement of a series between two times (None if unavailable)."""
        if s is None:
            return None
        i0, i1 = at(s, t0), at(s, t1)
        if i0 is None or i1 is None:
            return None
        return math.hypot(s[1][i1] - s[1][i0], s[2][i1] - s[2][i0])

    print(f"{'t_rel':>7} {'SE step':>8} {'dt_ms':>6} {'SE yaw':>7} | {'raw aft_mapped':>14} {'laser_odom':>10} {'utlidar_odom':>12} | plane_ok  trans_last  iters | health")
    t0 = t[0]
    counts = {"raw_jumped_too": 0, "fusion_only": 0, "ref_disagrees": 0}
    for i in jumps[:60]:
        ta, tb = t[i], t[i + 1]
        dr = disp(raw, ta - 0.05, tb + 0.05)
        dl = disp(lo, ta - 0.05, tb + 0.05)
        du = disp(ref, ta, tb)
        si = bisect.bisect_right(stats_t, tb) - 1
        st = stats[si][1] if si >= 0 else None
        hi = [h[1].data for h in health if ta - 0.5 <= h[0] <= tb + 0.5]
        s_st = f"{st.plane_match_success:6d}  {st.translation_from_last:9.3f}  {st.n_iterations:5d}" if st else "   n/a"
        f = lambda v: "   n/a" if v is None else f"{v:6.2f}m"
        print(f"{ta - t0:7.2f} {step[i]:7.2f}m {dt[i] * 1000:6.0f} {math.degrees(wrap(yaw[i + 1] - yaw[i])):+6.1f}d | "
              f"{f(dr):>14} {f(dl):>10} {f(du):>12} | {s_st} | {hi if hi else '-'}")
        if dr is not None and dr > 0.6 * step[i]:
            counts["raw_jumped_too"] += 1
        elif dr is not None:
            counts["fusion_only"] += 1
        if du is not None and du < 0.3 * step[i]:
            counts["ref_disagrees"] += 1
    print("\nsummary:", counts, "(raw_jumped_too = jump already present in lidar mapping;"
          " fusion_only = introduced by imu_preintegration; ref_disagrees = robot itself did not move that much)")

    fails = [h for h in health if not h[1].data]
    print(f"health=false messages: {len(fails)} / {len(health)}")
    if stats:
        pm = np.array([r[1].plane_match_success for r in stats])
        print(f"plane_match_success: min {pm.min()} p5 {np.percentile(pm, 5):.0f} median {np.median(pm):.0f}")


if __name__ == "__main__":
    main()
