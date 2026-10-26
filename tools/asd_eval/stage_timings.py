#!/usr/bin/env python3
"""Aggregate existing stage logs for one sequential golden run, without saving raw logs."""

import argparse
import collections
import datetime
import json
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="evaluation JSON with revision and timestamps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", help="UTC start timestamp for older runs without started_at")
    args = parser.parse_args()
    with open(args.run) as stream:
        config = json.load(stream)["config"]
    revision = config["revision"]
    start = args.start or config["started_at"]
    # Allow a small delay in ingesting final stdout lines.
    end = datetime.datetime.fromisoformat(config["generated_at"].replace("Z", "+00:00"))
    end += datetime.timedelta(seconds=2)
    query = (f'resource.type="cloud_run_revision" '
             f'AND resource.labels.revision_name="{revision}" '
             f'AND timestamp>="{start}" AND timestamp<="{end.isoformat()}"')
    result = subprocess.run([
        "gcloud", "logging", "read", query, "--project", "biometry-416410",
        "--order", "asc", "--limit", "10000", "--format", "json",
    ], check=True, text=True, capture_output=True)
    stages = collections.defaultdict(list)
    for entry in json.loads(result.stdout):
        message = entry.get("jsonPayload", {}).get("msg", entry.get("textPayload", ""))
        match = re.fullmatch(r"\s*(DL|VP|IP|FD|FC|VE|VI):\s*([0-9.eE+-]+)s?\s*", message)
        if match:
            stages[match[1]].append(float(match[2]))
    # VP is cumulative from the download start, unlike the other stages.
    if len(stages["VP"]) == len(stages["DL"]):
        stages["transcode"] = [v - d for v, d in zip(stages["VP"], stages["DL"])]
    summary = {stage: {"n": len(values), "mean_s": statistics.mean(values),
                       "median_s": statistics.median(values)}
               for stage, values in stages.items() if values}
    payload = {"run": args.run, "revision": revision, "start": start,
               "end": end.isoformat(), "stages": summary}
    with open(args.out, "w") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))
    if not summary:
        raise SystemExit("No stage timings found")


if __name__ == "__main__":
    main()
