# Raw preprocessing: labeled GPU comparison — 2026-10-21

The exact production image and the raw-preprocessing image each processed the
same frozen 50 human-labeled clips in isolated GPU staging, with zero request
errors. Production traffic was not used. These results belong to the images
recorded in `production.json` and `raw.json`, before the PR cleanup.

## Corpus and decision rule

- Main set: 48 clips, comprising 34 speaking and 14 not speaking.
- Two synchronized deepfake positives are reported separately as robustness
  checks. Both versions correctly classify both clips as speaking.
- Labels cover the first 10 seconds, or the whole clip when shorter. Three
  uncertain face-visibility labels retain their explicit speaking labels.
- 36 entries remain unlabeled; four entries removed by the annotator are not
  reintroduced. `labels.json` retains the original annotation SHA-256.
- API code 0 means speaking; codes 1 and 2 count as binary-negative decisions.
  Code 2 means no usable face track and is also counted separately.
- `manifest.json` freezes object names/order; GCS object generations are not pinned.

## Main-set quality

| Version | Accuracy | Precision | Recall | F1 | TP / TN / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| Production | 95.83% | 94.44% | 100% | 97.14% | 34 / 12 / 2 / 0 |
| Raw | 93.75% | 91.89% | 100% | 95.77% | 34 / 11 / 3 / 0 |

Across all 50 clips, accuracy is 96% for production and 94% for raw.

- `clip_24_own_video.webm` and `clip_35_own_video.webm` are shared false positives.
- `clip_47_own_video.webm` is the additional raw false positive: production
  returns code 1 / result 1.06 / score 5.30; raw returns code 0 / result 4.26 /
  score 21.30. The speaking cutoff remains 4%.
- `clip_40_own_video.webm` changes from code 2 to code 1. Both decisions are
  correctly binary-negative; raw obtains a usable face track.
- `clip_36_own_video.webm` has a human no-face label, but both versions return
  code 1. The speaking decision is correct in both cases.

Raw has 42 full parity passes, six additional numeric outliers and two API-code
changes versus production under the original absolute ±2.0 comparison gate.
Only one change affects the binary speaking decision. The gate is retained as
a diagnostic and was not widened to accept raw.

## HTTP latency, all 50 clips

| Version | Mean | Median | First request |
| --- | ---: | ---: | ---: |
| Production | 5.528 s | 5.569 s | 16.278 s |
| Raw | 2.910 s | 2.706 s | 14.759 s |

The observed median reduction is 51.4% versus production. Requests were serial
within each revision; different revisions ran concurrently. HTTP timings
include GCS download, network and cold starts. These are observed sample
latencies, not a controlled production throughput benchmark.

## Accepted trade-off and limits

The task owner accepted the additional false positive for the preprocessing
speed improvement. No additional false negatives occurred in this set. Only
14 main clips are negative: one error changes their measured false-positive
rate from 14.29% to 21.43%. A single annotator and related recordings limit
population-level conclusions. No thresholds or model weights were changed.

The PR selects raw preprocessing directly and preserves the evaluated media
algorithm. A fresh GPU run of the final image is still required before rollout.
Concurrent request safety and the remaining cropped-face XVID roundtrip are
separate work. Rollback uses a previously evaluated immutable image/revision.

Recompute classification metrics and per-clip changes from the repository root:

```bash
python3 tools/asd_eval/accuracy.py \
  --labels tools/asd_eval/results/labeled_20261021/labels.json \
  --run production=tools/asd_eval/results/labeled_20261021/production.json \
  --run raw=tools/asd_eval/results/labeled_20261021/raw.json \
  --out /tmp/bm494-metrics.json
```

The
[full experiment report](https://gravityfusion.atlassian.net/wiki/spaces/B/pages/4212621315)
contains the conservative/hybrid comparisons and the complete evidence archive.
