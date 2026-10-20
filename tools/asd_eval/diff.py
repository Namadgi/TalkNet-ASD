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
    python3 diff.py baseline.json candidate.json [--result-tol 1.0 --score-tol 1.0]
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {r["clip"]: r for r in data["results"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--result-tol", type=float, default=1.0)
    ap.add_argument("--score-tol", type=float, default=1.0)
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    fails, flags, oks = [], [], []
    for clip, b in base.items():
        c = cand.get(clip)
        if c is None:
            fails.append((clip, "missing in candidate"))
            continue
        if b["error"] or c["error"]:
            if c["error"] and not b["error"]:
                fails.append((clip, f"new error: {c['error']}"))
            continue
        if b["code"] != c["code"]:
            fails.append((clip, f"code {b['code']} -> {c['code']}"))
            continue
        dr = abs((c["result"] or 0) - (b["result"] or 0))
        ds = abs((c["score"] or 0) - (b["score"] or 0))
        if dr > args.result_tol or ds > args.score_tol:
            flags.append((clip, f"result Δ{dr:.2f} score Δ{ds:.2f}"))
        else:
            oks.append(clip)

    extra = set(cand) - set(base)

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
    if extra:
        print(f"\nnote: {len(extra)} clips only in candidate (ignored): "
              f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    if fails:
        sys.exit(1)
    if flags:
        sys.exit(2)
    print("\nOK - within tolerance")
    sys.exit(0)


if __name__ == "__main__":
    main()
