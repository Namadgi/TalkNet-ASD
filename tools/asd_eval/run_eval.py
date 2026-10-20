#!/usr/bin/env python3
"""
ASD golden-set runner.

Drives a deployed ml-asd service over HTTP against a GCS clip corpus and records
the ResponseModel output per clip. Runs anywhere (no local GPU needed) - the
model runs on the Cloud Run service; this is just an HTTP client.

Default mode = "gcs": calls /predictions/mono (the production code path). The
service downloads each clip from GCS itself using a short-lived access token, so
no local download is needed. Mode "upload" POSTs the file to /upload instead
(note: /upload can 500 on newer FastAPI builds due to an UploadFile isinstance
check in the service - prefer gcs).

Usage:
    python3 run_eval.py \
        --endpoint https://ml-asd-golden-xxxx-uc.a.run.app \
        --bucket biometry-test --prefix bm472-eval/clips/ \
        --out results/baseline.json [--auth] [--limit N]

Output JSON:
    {
      "config": {...},
      "results": [
        {"clip", "code", "result", "score", "description", "latency_s", "error"},
        ...
      ]
    }
"""
import argparse
import json
import os
import subprocess
import sys
import time

import requests

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".mkv")


def sh(cmd):
    """Run a command, return stdout, raise on failure."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def list_objects(bucket, prefix, project):
    """Return (object_name, clip_filename) for every video under the prefix."""
    out = sh(["gcloud", "storage", "ls", f"gs://{bucket}/{prefix}**", "--project", project])
    objs = []
    for line in out.splitlines():
        line = line.strip()
        if line.lower().endswith(VIDEO_EXTS):
            object_name = line[len(f"gs://{bucket}/"):]  # strip gs://bucket/
            objs.append((object_name, line.split("/")[-1]))
    objs.sort()
    return objs


def identity_token():
    return sh(["gcloud", "auth", "print-identity-token"]).strip()


def access_token():
    return sh(["gcloud", "auth", "print-access-token"]).strip()


def call_predict(endpoint, bucket, object_name, acc_tok, headers, timeout):
    url = endpoint.rstrip("/") + "/predictions/mono"
    body = {"instances": [{"token": acc_tok, "bucket_name": bucket,
                           "object_name": object_name}]}
    t0 = time.time()
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    return r.json(), latency


def call_upload(endpoint, bucket, object_name, project, cache, headers, timeout):
    url = endpoint.rstrip("/") + "/upload"
    name = object_name.split("/")[-1]
    dest = os.path.join(cache, name)
    if not os.path.isfile(dest):
        sh(["gcloud", "storage", "cp", f"gs://{bucket}/{object_name}", dest,
            "--project", project])
    with open(dest, "rb") as f:
        files = {"file": (name, f, "application/octet-stream")}
        t0 = time.time()
        r = requests.post(url, files=files, headers=headers, timeout=timeout)
        latency = time.time() - t0
    r.raise_for_status()
    return r.json(), latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="service base URL")
    ap.add_argument("--bucket", default="biometry-test")
    ap.add_argument("--prefix", default="bm472-eval/clips/")
    ap.add_argument("--project", default="biometry-416410")
    ap.add_argument("--out", required=True, help="output results JSON path")
    ap.add_argument("--mode", choices=["gcs", "upload"], default="gcs",
                    help="gcs=/predictions/mono (prod path); upload=/upload")
    ap.add_argument("--cache", default="clips_cache", help="local dir (upload mode)")
    ap.add_argument("--auth", action="store_true", help="send gcloud identity token")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (0=all)")
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout s")
    ap.add_argument("--name", default="", help="run label stored in config")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    headers = {}
    if args.auth:
        headers["Authorization"] = f"Bearer {identity_token()}"
    acc_tok = access_token() if args.mode == "gcs" else None

    objs = list_objects(args.bucket, args.prefix, args.project)
    if args.limit:
        objs = objs[: args.limit]
    print(f"{len(objs)} clips from gs://{args.bucket}/{args.prefix} (mode={args.mode})",
          file=sys.stderr)

    results = []
    for i, (object_name, name) in enumerate(objs, 1):
        rec = {"clip": name, "code": None, "result": None, "score": None,
               "description": None, "latency_s": None, "error": None}
        try:
            if args.mode == "gcs":
                body, latency = call_predict(args.endpoint, args.bucket, object_name,
                                             acc_tok, headers, args.timeout)
            else:
                body, latency = call_upload(args.endpoint, args.bucket, object_name,
                                            args.project, args.cache, headers,
                                            args.timeout)
            rec.update(code=body.get("code"), result=body.get("result"),
                       score=body.get("score"), description=body.get("description"),
                       latency_s=round(latency, 4))
            print(f"[{i}/{len(objs)}] {name}: code={rec['code']} "
                  f"result={rec['result']} score={rec['score']} ({latency:.2f}s)",
                  file=sys.stderr)
        except Exception as e:
            detail = ""
            if isinstance(e, requests.HTTPError) and e.response is not None:
                detail = f" | {e.response.text[:200]}"
            rec["error"] = f"{e}{detail}"
            print(f"[{i}/{len(objs)}] {name}: ERROR {rec['error']}", file=sys.stderr)
        results.append(rec)

    payload = {
        "config": {
            "name": args.name or os.path.splitext(os.path.basename(args.out))[0],
            "endpoint": args.endpoint,
            "mode": args.mode,
            "bucket": args.bucket,
            "prefix": args.prefix,
            "n_clips": len(results),
            "n_errors": sum(1 for r in results if r["error"]),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}  (errors: {payload['config']['n_errors']})", file=sys.stderr)


if __name__ == "__main__":
    main()
