#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""Per-10-second frame rate and what the slow frames have in common, from an OOT_FRAME_CSV file (runtime/prof_win.c).

    .venv-win\\Scripts\\python.exe tools\\frame_csv.py build/logs/frames.csv [--slow 25]

Run the game with --trace perf to fill the gpu column. "game" is the frame interval minus everything the render
thread is known to have done: the game's own CPU work, and any wait not named.
"""
import argparse
import csv
import statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--slow", type=float, default=25.0)
ap.add_argument("--bin", type=float, default=10.0)
a = ap.parse_args()
rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(a.csv))]
rows = [r for r in rows if r["ms"] > 0]
for r in rows:
    r["game"] = max(0.0, r["ms"] - r["replay"] - r["pace"] - r["fence"] - r["drain"])
bins = defaultdict(list)
for r in rows:
    bins[int(r["t_s"] // a.bin)].append(r)
print(f"{'time':>9} {'fps':>6} {'1%low':>6} {'>25ms':>6} {'>50ms':>6} {'tex MiB':>8} {'tex #':>6} {'pipe ms':>8} {'vram GiB':>8}")
for b, rr in sorted(bins.items()):
    t = sum(x["ms"] for x in rr) / 1000
    ms = sorted(x["ms"] for x in rr)
    low = ms[-max(1, len(ms) // 100)]
    print(f"{b * a.bin:6.0f}-{(b + 1) * a.bin:<4.0f}{len(rr) / t if t else 0:6.1f} {1000 / low:6.1f} {sum(m > 25 for m in ms):6d} {sum(m > 50 for m in ms):6d}"
          f" {sum(x['tex_mib'] for x in rr):8.1f} {sum(x['tex_new'] + x['tex_re'] for x in rr):6.0f} {sum(x['pipe'] for x in rr):8.0f} {max(x.get('vram_mib', 0) for x in rr) / 1024:8.2f}")
fast = [r for r in rows if r["ms"] <= a.slow]
slow = [r for r in rows if r["ms"] > a.slow]
print(f"\nframes {len(rows)}, over {a.slow:.0f} ms: {len(slow)} ({100 * len(slow) / len(rows):.1f}%), of the time {100 * sum(r['ms'] for r in slow) / sum(r['ms'] for r in rows):.1f}%")
cols = ["ms", "replay", "tex", "tex_alloc", "tex_swizzle", "astc", "pipe", "xlate", "fence", "drain", "pace", "gpu", "game", "tex_new", "tex_re", "tex_mib"]
print(f"{'':12s}{'median fast':>12}{'median slow':>12}{'mean slow':>11}")
for c in cols:
    print(f"{c:12s}{st.median(r[c] for r in fast) if fast else 0:12.2f}{st.median(r[c] for r in slow) if slow else 0:12.2f}{st.mean(r[c] for r in slow) if slow else 0:11.2f}")
if slow:
    # each slow frame's largest named part
    cause = defaultdict(int)
    for r in slow:
        parts = {"texture": r["tex"], "pipeline": r["pipe"] + r["xlate"], "fence/GPU": r["fence"] + r["drain"], "replay other": max(0, r["replay"] - r["tex"] - r["pipe"] - r["xlate"]), "game CPU": r["game"], "pace": r["pace"]}
        cause[max(parts, key=parts.get)] += 1
    print("\nlargest part of each slow frame:", dict(sorted(cause.items(), key=lambda x: -x[1])))
