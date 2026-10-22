# ASD preprocessing evaluation (BM-494)

The serving pipeline decodes and normalizes source media once with FFmpeg,
streaming raw YUV420p video and mono PCM16 audio through a timestamped NUT pipe.
PyAV converts the video to RGB arrays without a normalized MP4 encode/decode
roundtrip. Normalization keeps 640-pixel height, 25 fps and 16 kHz audio, with
video timestamps through 10.0 seconds inclusive and audio through the last
frame interval. The existing cropped-face XVID roundtrip remains.

Raw preprocessing is the only serving path in this branch; no `ASD_MEDIA_MODE`
setting is required. The conservative implementation and diagnostic hybrids
remain in the separate experiment branches, outside this PR.

## Accepted result and evidence

The [frozen labeled comparison](results/labeled_20261021/report.md) records 50
GPU requests per version: 48 main clips and two deepfake robustness checks.
Compared with the exact production image, raw median HTTP latency fell from
5.569 s to 2.706 s. Main-set accuracy changed from 95.83% to 93.75%: one additional
false positive and no additional false negatives. The task owner accepted this
trade-off. The 4% speaking cutoff and result-to-score mapping are unchanged.

Raw does **not** pass the old strict response-parity gate. `diff.py` retains that
gate (exact code, absolute result/score delta at most 2.0) as a diagnostic;
`accuracy.py` measures decisions against independent human labels instead.

The committed files preserve labels, the input manifest and the two original
response sets with immutable image/revision provenance. Intermediate experiment
runs and generated metric summaries are excluded. The complete experiment
history and evidence archive are in the
[Confluence report](https://gravityfusion.atlassian.net/wiki/spaces/B/pages/4212621315).
These GPU results predate the PR cleanup; rerun the final image before rollout.

## Local checks

Use a Python virtual environment and install FFmpeg on the host:

```bash
python3 -m pip install -r tools/asd_eval/requirements-test.txt
python3 -m unittest discover -s tools/asd_eval -p 'test_*.py'
bash -n tools/asd_eval/deploy_golden.sh
```

The media tests exercise real FFmpeg/PyAV decoding, timestamps, shapes, PCM
conversion and decoder failure/timeout cleanup. They do not load model weights
or run GPU inference. Exact codec controls should use the evaluated image's
versions; newer local codecs can behave differently.

Recompute the saved labeled metrics without a service or GPU:

```bash
python3 tools/asd_eval/accuracy.py \
  --labels tools/asd_eval/results/labeled_20261021/labels.json \
  --run production=tools/asd_eval/results/labeled_20261021/production.json \
  --run raw=tools/asd_eval/results/labeled_20261021/raw.json \
  --out /tmp/bm494-metrics.json
```

## Evaluate a candidate image

`run_eval.py` is an HTTP client. It calls `POST /predictions/mono` using the
service's GCS download path. The frozen manifest selects 50 clips from
`gs://biometry-test/bm472-eval/clips/`; it fixes object names/order, not object
generations. Do not replace those objects. Short-lived authentication tokens
are not stored in result files.

Build against an immutable dependency image to preserve codecs and weights.
The following command builds and deploys to the specified **staging** service:

```bash
bash tools/asd_eval/deploy_golden.sh ml-asd-golden \
  us-central1-docker.pkg.dev/biometry-416410/cloud-run-source-deploy/ml-asd-golden@sha256:d121d384aa0a5ae990545e7f9baa3b0386d1d3085c2b0464d756b6d8a87c3301
```

`Dockerfile.golden` overlays serving source on that image. The helper removes
any stale experiment mode from the staging revision. Omitting the image builds
from the repository Dockerfile, whose dependencies are not pinned. Use an
immutable image for comparisons.

```bash
python3 tools/asd_eval/run_eval.py --endpoint <STAGING_URL> --auth \
  --manifest tools/asd_eval/results/labeled_20261021/manifest.json \
  --revision <REVISION> --image <IMAGE_DIGEST> \
  --out tools/asd_eval/results/candidate.json
python3 tools/asd_eval/accuracy.py \
  --labels tools/asd_eval/results/labeled_20261021/labels.json \
  --run production=tools/asd_eval/results/labeled_20261021/production.json \
  --run candidate=tools/asd_eval/results/candidate.json \
  --out tools/asd_eval/results/candidate_metrics.json
python3 tools/asd_eval/diff.py \
  tools/asd_eval/results/labeled_20261021/raw.json \
  tools/asd_eval/results/candidate.json
```

`--limit N` runs a smoke subset; omit it for labeled accuracy. The evaluator
rejects incomplete, duplicate or invalid response sets. `diff.py` exits 0 for
parity, 2 for numeric differences and 1 for decision changes, invalid responses
or a different clip set. Compare with the saved raw run to check cleanup parity;
compare with production to inspect the accepted differences.

Requests run serially because the service still shares temporary filenames.
This change does not make concurrent requests safe. Keep the prior immutable
image for rollback; rollback uses a revision/image, not a mode switch.

## Timing

```bash
python3 tools/asd_eval/stage_timings.py tools/asd_eval/results/candidate.json \
  --out tools/asd_eval/results/candidate_timings.json
```

`DL` is GCS download; `VP` includes download and raw pipe processing; `IP` is
array assembly; `FD` is face detection; `FC` is track cropping; `VE` is feature
extraction; `VI` is TalkNet inference. Subtract `DL` from `VP` for normalization
time, and compare `VP - DL + IP` across different readers. No-face requests
omit later stages. Client HTTP latencies include network and cold starts and
are not a throughput or concurrency benchmark.
