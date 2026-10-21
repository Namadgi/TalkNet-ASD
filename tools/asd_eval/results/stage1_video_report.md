# BM-494: audio + video reuse evaluation, 2026-10-21

The combined candidate passes the full 90-clip GPU golden set: exact decision
codes and result/score differences within the user-approved ±2.0 points.
Video-frame reuse adds **no response drift** relative to the audio-only candidate.

## What changed

| Original operation | Final behavior |
|---|---|
| FFmpeg normalization into output.mp4 | Kept: 640h, 25fps, mono 16kHz |
| read_video decoding normalized video and audio | Kept; both arrays are reused |
| OpenCV full-source reread for face cropping | Removed; tracked RGB frames are converted to BGR in memory |
| FFmpeg audio crop into sync.wav | Removed; slice decoded PCM16 samples using 640 samples per video frame |
| FFmpeg mux into sync.avi | Removed; temp.avi is read directly |
| OpenCV decode of the cropped video | Kept; reads temp.avi |
| WAV file read before MFCC | Removed; MFCC receives the sliced array |

For tracked-face requests, full-source media reads drop from 4 to 2 and FFmpeg
subprocesses from 3 to 1. The cropped-face XVID encode/decode roundtrip remains.
The old list containing a second full set of decoded video frames is removed.
Concurrency changes and further in-memory-video work are deferred.

## Regression results

| Comparison | Decisions matching | Exact full responses | Largest result delta | Largest score delta | Gate |
|---|---:|---:|---:|---:|---|
| Original baseline -> combined | 90/90 | 84/90 | 1.48 | 0.82 | Pass |
| Audio-only -> combined | 90/90 | 90/90 | 0.00 | 0.00 | Pass |

There were no request errors. Decision counts remain 73 speaking, 8 not-speaking,
and 9 no-face. The six numeric changes are the same ones documented in
`stage1_audio_report.md`; no extra differences appeared after video reuse.

These are regression measurements against the existing model, not human-label
accuracy. The unchanged baseline was also repeated independently and matched
its original outputs exactly on all 90 clips. GCS object names/order are fixed
by the manifest; object contents are not generation-pinned.

## Performance

| Measurement | Original baseline | Audio-only | Combined |
|---|---:|---:|---:|
| Mean HTTP latency, 90 clips | 5.045s | 4.701s | 4.664s |
| Median HTTP latency, 90 clips | 4.785s | 4.585s | 4.511s |
| Mean FC stage, 81 tracked-face clips | 0.820s | 0.491s | 0.294s |

Combined vs original: mean HTTP latency is 7.6% lower and the FC stage is 64.2%
lower. Video reuse alone reduces the FC stage by 40.1% relative to audio-only;
its additional mean HTTP reduction in this run is 0.8%.

HTTP measurements include GCS download, network, and worker warm-up variability.
A separate 59.45s instance cold-start request was excluded from the full run and
saved locally as `video_warmup.json`. These are single sequential runs, not a
throughput or concurrency benchmark. FC baseline stage timings come from the
independent baseline repeat; HTTP baseline timings use the original saved run.
FC originally included full-video reread, face cropping, audio crop, and mux;
the final FC stage performs face cropping and array slicing. Transcoding remains
the largest stage (combined mean 2.190s).

## Provenance and artifacts

- GPU service: `ml-asd-golden`, project `biometry-416410`, region `us-central1`.
- Revision: `ml-asd-golden-00003-hjh`, 100% staging traffic during evaluation.
- Image: `us-central1-docker.pkg.dev/biometry-416410/cloud-run-source-deploy/ml-asd-golden@sha256:327e288a784ec8a0605cdbc2fbd0f0131c07825b0e977e7476e5fba9d3b209f2`.
- Cloud Build: `06e08742-7ef8-4310-91a3-f1f689ed59fe`.
- Baseline serving source: `f04513f` on `bm-438-ci`.
- Dependencies, codecs, and weights inherited from the exact baseline image via `Dockerfile.golden`.
- Full run: 2026-10-21 11:47:57–11:54:57 UTC, sequential GCS requests.
- Raw responses: `stage1_video.json`; stage summaries: `stage1_video_timings.json`.
- Local validation: 4 PCM conversion tests, 7 golden-gate tests, Python compilation, and whitespace/shell checks passed.

Production `ml-asd` was not deployed. The staging service remains available for
review and is configured to scale to zero when idle.
