# BM-494: audio reuse comparison, 2026-10-20

Status: evaluation complete; user approved increasing both result and score
tolerance to ±2.0 after reviewing the outliers. The audio candidate passes the
updated gate on 90/90 clips, with exact decision-code matching retained.

## Scope and provenance

The GPU staging service `ml-asd-golden` ran all 90 clips sequentially through
`/predictions/mono`. The candidate reuses `read_video` audio and removes the
second FFmpeg audio decode, WAV read, and audio/video mux. The video crop still
reads `output.mp4` using OpenCV and retains the XVID encode/decode path.

- Baseline source: `f04513f` (`bm-438-ci`), revision `ml-asd-golden-00001-5x2`.
- Baseline image digest: `sha256:d121d384aa0a5ae990545e7f9baa3b0386d1d3085c2b0464d756b6d8a87c3301`.
- Candidate revision: `ml-asd-golden-00002-rg5`.
- Candidate image digest: `sha256:cc5e3251162a4055a5df12522754ffa4729609603e9006167c722aebfab11c98`.
- Image repository: `us-central1-docker.pkg.dev/biometry-416410/cloud-run-source-deploy/ml-asd-golden`.
- Candidate image inherits the exact baseline dependencies and model weights.
- Candidate run: 18:11:04–18:18:08 UTC; no request errors.
- Production was not deployed. The separate CPU decoder diagnostic never started.

## Regression results

- Decision codes match on 90/90 clips: 73 speaking, 8 not-speaking, 9 no-face.
- All numeric outputs match exactly on 84/90 clips.
- Original ±1.0 gate: 86 pass, 4 numeric flags, 0 decision/error failures.
- Approved ±2.0 gate: 90 pass, 0 numeric flags, 0 decision/error failures.
- Maximum absolute result difference: 1.48 percentage points; mean: 0.0662.
- Maximum absolute score difference: 0.82 points; mean: 0.0369.
- An independent repeat of the baseline had zero numeric drift on all 90 clips.

All six clips with numeric changes are listed below. All remain code 0 (speaking).

| Clip | Baseline result | Candidate result | Absolute result delta | Absolute score delta | ±1.0 gate |
|---|---:|---:|---:|---:|---|
| clip_11_qJF25x.webm | 78.74 | 79.89 | 1.15 | 0.64 | Flag |
| clip_20_own_IMG_3866.mov | 87.01 | 85.71 | 1.30 | 0.72 | Flag |
| clip_55_own_video.webm | 49.41 | 50.00 | 0.59 | 0.33 | Pass |
| clip_62_bench_anna.mp4 | 75.92 | 76.96 | 1.04 | 0.58 | Flag |
| clip_66_df_1.mp4 | 86.40 | 86.80 | 0.40 | 0.23 | Pass |
| clip_80_df_455513.mp4 | 99.26 | 97.78 | 1.48 | 0.82 | Flag |

The speaking/not-speaking threshold is 4% result. All changed clips are far
from that threshold; the lowest result among them is 49.41%. This corpus does
not establish that future clips near the threshold will preserve decisions.
There are no human ASD labels, so these are regression metrics, not accuracy.

## Timing results

| Measurement | Baseline | Candidate | Observed change |
|---|---:|---:|---:|
| Mean HTTP latency, same 90 clips | 5.045s | 4.701s | 6.8% lower |
| Median HTTP latency, same 90 clips | 4.785s | 4.585s | 4.2% lower |
| Mean FC stage, 81 tracked-face clips | 0.820s | 0.491s | 40.2% lower |

HTTP timings compare the original committed baseline with the candidate and
include GCS download/network overhead. Stage timings compare the separate
baseline repeat with the candidate; FC includes cropping plus audio crop/mux
in the old path. These are single-run observations, not a throughput benchmark.

## Artifacts and decision

Raw outputs: `baseline.json`, `baseline_repeat.json`, `stage1_audio.json`.
Timing summaries: `baseline_repeat_timings.json`, `stage1_audio_timings.json`.

The cause of numeric drift has not yet been isolated. After reviewing the full
responses, the user accepted the observed differences and approved ±2.0 for
both numeric fields. Decision codes must still match exactly. The table above
preserves the original ±1.0 flags for reference; all six rows pass ±2.0.
