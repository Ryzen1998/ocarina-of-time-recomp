#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""The account of a frame's GPU time from an OOT_GPU_ACCT file (runtime/vk_gpu_acct.c).

    .venv-win\\Scripts\\python.exe tools\\gpu_acct.py build/logs/gpu-acct.csv [--top 12]

Per window: median per frame of the time from one present to the next (the frame), of the time the queue had work
(the union of the command buffers), of each category inside them, and the time the queue stood idle between them.
"Replay, other" is what the replay's command buffers hold outside anything stamped: barriers, the setup of the mesh
stages (fills, small copies) and any waiting inside them. Times inside a command buffer are its own stamps, so they
add up to the part only as far as the stamps cover it.
"""
import argparse
import collections
import csv
import statistics

CATS = ["part", "mirror up", "mirror down", "upload", "present", "pass", "compute", "mesh: task stage", "mesh: mesh stage", "mesh: compact",
        "copy/clear"]
PART, UP, DOWN, UPLOAD, PRESENT, PASS, CS, TASK, MESH, COMPACT, XFER = range(11)


def union(iv):
    t, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            t += e - s
            end = e
        elif e > end:
            t += e - end
            end = e
    return t


def pass_kind(r):
    wh, b = r["a"], r["b"]
    w, h, draws, mesh, nc, depth = wh >> 16, wh & 0xffff, b >> 32, (b >> 16) & 0xffff, (b >> 8) & 0xff, b & 1
    kind = f"{w}x{h}"
    kind += f" c{nc}" if nc else " depth only"
    if nc and depth:
        kind += "+d"
    kind += f" fmt{r['id']:x}" if nc else ""
    return kind, draws, mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()
    wins = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(a.csv) as f:
        for row in csv.DictReader(f):
            r = {"cat": int(row["cat"]), "id": int(row["id"], 16), "a": int(row["a"]), "b": int(row["b"]), "s": int(row["start_ns"]) / 1e6,
                 "e": int(row["end_ns"]) / 1e6}
            wins[int(row["window"])][int(row["frame"])].append(r)
    for w, frames in sorted(wins.items()):
        fr = sorted(frames)[1:-1]
        presents = {f: max(r["e"] for r in frames[f] if r["cat"] == PRESENT) for f in fr if any(r["cat"] == PRESENT for r in frames[f])}
        # the rows of a frame number are from one present to the next (the counter moves at the present)
        rows = []
        for f in fr:
            if f - 1 not in presents or f not in presents:
                continue
            per = collections.Counter()
            cnt = collections.Counter()
            evs = [r for r in frames[f]]
            top = [(r["s"], r["e"]) for r in evs if r["cat"] in (PART, UP, UPLOAD, PRESENT)]
            frame_t = presents[f] - presents[f - 1]
            busy = union(top)
            for r in evs:
                per[r["cat"]] += r["e"] - r["s"]
                cnt[r["cat"]] += 1
            inside = per[PASS] + per[CS] + per[TASK] + per[MESH] + per[COMPACT] + per[XFER] + per[DOWN]
            per["other"] = per[PART] - inside
            per["frame"] = frame_t
            per["busy"] = busy
            per["idle"] = frame_t - busy
            rows.append((f, per, cnt, evs))
        if not rows:
            print(f"window {w}: no complete frames")
            continue
        med = lambda k: statistics.median(p[k] for _, p, _, _ in rows)
        cmed = lambda k: statistics.median(c[k] for _, _, c, _ in rows)
        print(f"\n=== window {w}: frames {rows[0][0]}..{rows[-1][0]}, {len(rows)} complete; medians, ms per frame")
        print(f"frame period {med('frame'):.2f}, queue busy {med('busy'):.2f}, queue idle {med('idle'):.2f}")
        for c, n in [(PASS, "render passes"), (CS, "game compute"), (TASK, "mesh: task stage"), (MESH, "mesh: mesh stage"), (COMPACT, "mesh: compact"),
                     (XFER, "copies/clears"), (DOWN, "mirror down (write back)"), ("other", "replay, other (barriers, mesh setup, waits)"),
                     (UP, "mirror up"), (UPLOAD, "upload batches"), (PRESENT, "present")]:
            nn = f" x{cmed(c):.0f}" if c != "other" else ""
            print(f"  {n:45s} {med(c):7.2f}{nn}")
        for c, n in [(UP, "mirror up"), (DOWN, "mirror down")]:
            mib = statistics.median(sum(r["a"] for r in evs if r["cat"] == c) for *_, evs in rows) / 2**20
            t = med(c)
            print(f"  {n}: {mib:.1f} MiB per frame, {mib / t:.1f} MiB/ms" if t > 0 else f"  {n}: none")
        # passes and programs
        passes = collections.defaultdict(lambda: [0.0, 0, 0, 0])
        cs = collections.defaultdict(lambda: [0.0, 0, 0])
        for _, p, _, evs in rows:
            for r in evs:
                if r["cat"] == PASS:
                    k, d, m = pass_kind(r)
                    e = passes[k]
                    e[0] += r["e"] - r["s"]; e[1] += 1; e[2] += d; e[3] += m
                elif r["cat"] in (CS, TASK, MESH):
                    e = cs[(r["cat"], r["id"])]
                    e[0] += r["e"] - r["s"]; e[1] += 1; e[2] += r["a"]
        n = len(rows)
        print(f"\n  passes by target (mean ms per frame, passes, draws, of which mesh draws):")
        for k, e in sorted(passes.items(), key=lambda x: -x[1][0])[:a.top]:
            print(f"    {k:32s} {e[0] / n:7.2f} ms  x{e[1] / n:5.1f}  draws {e[2] / n:6.1f}  mesh {e[3] / n:5.1f}")
        print(f"\n  programs (mean ms per frame, dispatches):")
        for (c, i), e in sorted(cs.items(), key=lambda x: -x[1][0])[:a.top]:
            print(f"    {CATS[c]:18s} {i:016x} {e[0] / n:7.2f} ms  x{e[1] / n:5.1f}")


if __name__ == "__main__":
    main()
