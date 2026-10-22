#!/usr/bin/env python3
"""
Compare two ASD eval runs (baseline vs candidate).

Gate:
    code   : exact match (categorical decision: speaking / not / no-face)
    result : abs(delta) <= --result-tol
    score  : abs(delta) <= --score-tol

Exit code:
    0  all pass
    1  any FAIL (code flip, new error, or missing clip)
    2  no FAIL but some FLAG (numeric drift beyond tolerance)

Usage:
    python3 diff.py baseline.json candidate.json [--result-tol 2.0 --score-tol 2.0]
"""
import argparse
import json
import math
import statistics
import sys


def load(path):
    with open(path) as f:
        data = json.load(f)
    rows = data["results"]
    if not rows or len({r["clip"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: empty results or duplicate clip names")
    return {r["clip"]: r for r in rows}


def valid(row):
    return (not row.get("error") and type(row.get("code")) is int
            and row["code"] in (0, 1, 2)
            and all(type(row.get(k)) in (int, float)
                    and math.isfinite(row[k]) and 0 <= row[k] <= 100
                    for k in ("result", "score")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--result-tol", type=float, default=2.0)
    ap.add_argument("--score-tol", type=float, default=2.0)
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    fails, flags, oks = [], [], []
    for clip, b in base.items():
        c = cand.get(clip)
        if c is None:
            fails.append((clip, "missing in candidate"))
            continue
        if not valid(b) or not valid(c):
            fails.append((clip, "invalid response or request error in baseline/candidate"))
            continue
        if b["code"] != c["code"]:
            fails.append((clip, f"code {b['code']} -> {c['code']}"))
            continue
        dr = abs(c["result"] - b["result"])
        ds = abs(c["score"] - b["score"])
        result_exceeds = dr > args.result_tol and not math.isclose(
            dr, args.result_tol, rel_tol=0, abs_tol=1e-9)
        score_exceeds = ds > args.score_tol and not math.isclose(
            ds, args.score_tol, rel_tol=0, abs_tol=1e-9)
        if result_exceeds or score_exceeds:
            flags.append((clip, f"result Δ{dr:.2f} score Δ{ds:.2f}"))
        else:
            oks.append(clip)

    extra = set(cand) - set(base)
    fails.extend((clip, "extra clip in candidate") for clip in sorted(extra))

    print(f"clips: {len(base)} baseline | pass {len(oks)} | "
          f"flag {len(flags)} | fail {len(fails)}")
    if flags:
        print("\nFLAG (numeric drift > tol):")
        for clip, msg in flags:
            print(f"  {clip}: {msg}")
    if fails:
        print("\nFAIL:")
        for clip, msg in fails:
            print(f"  {clip}: {msg}")
    pairs = [(b["latency_s"], cand[clip]["latency_s"]) for clip, b in base.items()
             if clip in cand and valid(b) and valid(cand[clip])
             and b.get("latency_s") is not None and cand[clip].get("latency_s") is not None]
    if pairs:
        before, after = zip(*pairs)
        print(f"\nHTTP latency ({len(pairs)} paired clips, includes download/network):")
        print(f"  mean: {statistics.mean(before):.3f}s -> {statistics.mean(after):.3f}s")
        print(f"  median: {statistics.median(before):.3f}s -> {statistics.median(after):.3f}s")

    if fails:
        sys.exit(1)
    if flags:
        sys.exit(2)
    print("\nOK - within tolerance")
    sys.exit(0)


if __name__ == "__main__":
    main()
