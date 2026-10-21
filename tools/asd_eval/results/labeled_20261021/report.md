# Human-labeled ASD pipeline comparison — 2026-10-21

All three versions were freshly evaluated on GPU against the same frozen 50 human-labeled GCS clips. The exact production image ran in isolated staging; production traffic was not used or changed. No thresholds, model weights, or serving source were changed during this evaluation.

## Corpus

- 48 main clips: 34 speaking, 14 not speaking.
- 2 known deepfake clips (`clip_65_df_s.mp4`, `clip_80_df_455513.mp4`) scored separately as robustness checks; both labeled speaking.
- 36 entries remain unlabeled; 4 files removed by the annotator are not reintroduced.
- Three face-visibility labels are uncertain, but their explicit binary speaking labels are included. They are excluded from the separate known-face visibility diagnostic.
- The original annotation file is unchanged. `labels.json` is its frozen, normalized snapshot, with the source SHA-256. `manifest.json` fixes the exact input objects.
- Predictions are speaking iff API `code == 0`; codes 1 and 2 are negative binary decisions. Code 2 is also reported separately, never silently discarded.

## Main-set classification (48 clips)

| Version | Accuracy | Precision | Recall | F1 | TP / TN / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| production | 95.83% | 94.44% | 100.00% | 97.14% | 34 / 12 / 2 / 0 |
| stage1 | 95.83% | 94.44% | 100.00% | 97.14% | 34 / 12 / 2 / 0 |
| raw | 93.75% | 91.89% | 100.00% | 95.77% | 34 / 11 / 3 / 0 |

All-50 accuracy (including the separate deepfake checks): production 96%, stage1 96%, raw 94%. Both deepfake checks pass in all versions. None of the versions miss a labeled speaking clip.

## Disagreements

- `clip_24_own_video.webm` and `clip_35_own_video.webm`: human label is not speaking; all three versions incorrectly predict speaking.
- `clip_47_own_video.webm`: human label is not speaking. Production and stage1 return code 1 / result 1.06 / score 5.3. Raw returns code 0 / result 4.26 / score 21.3: one additional false positive.
- `clip_40_own_video.webm`: production/stage1 return code 2, raw code 1. All are binary-negative and agree with the human speaking label. The face label is yes; raw improves this particular usable-face outcome.
- `clip_36_own_video.webm`: human face label is no, but every version returns code 1 rather than code 2. Review if exact no-face behavior matters; binary speaking classification remains correct.

## Response parity versus the exact production image

- Stage1: 50/50 code matches and every result/score within absolute ±2.0.
- Raw: 42 within the full parity gate, 6 additional numeric outliers, and 2 three-way API-code changes (only one binary speaking change).
- Accuracy and numeric response parity are different requirements. This test does not justify changing either threshold automatically.

## HTTP timing (all 50 clips)

| Version | Mean | Median | Maximum |
| --- | ---: | ---: | ---: |
| production | 5.528 s | 5.569 s | 16.278 s |
| stage1 | 5.945 s | 5.223 s | 49.483 s |
| raw | 2.910 s | 2.706 s | 14.759 s |

Timings include GCS download, network, and cold starts. First-request latencies were approximately 16.28 s (production), 49.48 s (stage1), and 14.76 s (raw), so unadjusted means are not clean steady-state comparisons. Requests are serial within each isolated revision; different revisions were evaluated concurrently. Treat these timings as directional, not a controlled production throughput benchmark.

## Interpretation

The validated bm-494-decode-once image retains production binary decisions on this set and passes the existing ±2.0 numeric gate. Raw preprocessing is faster but adds one false positive. With only 14 negative main clips, that one error changes negative-class false-positive rate from 14.29% to 21.43%; it is a measured regression on this corpus, not proof of a population-wide rate. Inspect clip 47 before deciding whether to accept this trade-off. No merge, push, PR, or production deployment was performed.

Full image/revision provenance and per-clip responses are in `production.json`, `stage1.json`, `raw.json`; classification details are in `metrics.json`.

