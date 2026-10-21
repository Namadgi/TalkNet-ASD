# BM-494: raw-media experiment, 2026-10-21

Verdict: substantially faster, but **do not merge**. The full 90-clip GPU run
fails the agreed gate (exact decision codes, result/score within ±2.0 points).
The validated `bm-494-decode-once` branch is unchanged. No PR was opened.

## Implementation

Experimental branch: `bm-494-raw-preprocessing`, branched from `6637385`.
Serving-source commit: `c754eb6`.

Instead of encoding `output.mp4` and decompressing it with `read_video`, FFmpeg
decodes the original input and normalizes video/audio once. Uncompressed
YUV420P video plus mono PCM16 audio travel over a timestamped NUT stdout pipe.
PyAV unpacks the raw frames and converts video to RGB arrays. This is not a
second compressed-media decode. Normalization remains 640 pixels high, 25fps,
16kHz mono. Audio timestamp gaps are represented with silence. Constant frame
rate is explicit because the raw container has different muxer defaults.

The experimental normalizer stops at 10.08s, retaining video timestamps through
10.0s inclusive and up to 10.04s of audio for the final video's frame interval.
The previous normalizer processed the whole source before `read_video` applied
its 10s window. Thus this experiment combines removal of the compressed
roundtrip with avoiding unused long-clip normalization; their separate speed
contributions have not been measured. Exact per-clip sample/frame parity with
the previous AAC/MP4 window handling is not established.

The cropped-face XVID `temp.avi` encode/decode roundtrip and model code remain
unchanged. This experiment is not a claim that the entire pipeline has only
one decode operation. Concurrency changes remain deferred.

## Performance

| Measurement | Validated Stage 1 | Raw experiment |
|---|---:|---:|
| Mean client HTTP latency, 90 clips | 4.664s | 2.413s |
| Median client HTTP latency | 4.511s | 2.464s |
| Mean normalization + media-array preparation | 2.999s | 0.810s |
| Mean face detection | 0.362s | 0.345s |

Observed mean HTTP reduction: 48.3% versus validated Stage 1, 52.2% versus the
original 5.045s baseline. The normalization/media preparation reduction is
73.0%. The raw preparation total combines pipe normalization/unpacking (0.692s)
and array assembly (0.117s); the old total combines transcode (2.190s) and
`read_video` (0.808s). Individual VP/IP stages are not directly comparable
because normalization and pipe consumption now overlap.

The five-clip smoke test preceded the full run and is excluded from these
numbers. Requests were sequential, not a throughput/concurrency benchmark.
Dependencies, codecs, model weights and GPU resources were held fixed using
the immutable baseline container. Runs were not interleaved, so network/GCS
and runtime variance remain; the raw run's mean GCS download was 0.175s versus
0.218s for Stage 1. This is a strong observed speedup, not a production SLO.

## Quality gate

- 90/90 requests succeeded.
- Decisions match on 88/90 clips; two changed.
- 12 additional clips exceed the numeric ±2.0 gate.
- 76/90 clips pass the complete gate; 40/90 responses are exactly unchanged
  relative to validated Stage 1.
- Largest result difference: 7.75 points. Largest score difference: 16.00.
- The original-baseline comparison also has 76 passes, 12 flags, two failures.

| Clip | Code old → raw | Result old → raw | Score old → raw |
|---|---|---|---|
| clip_21_own_Movie_on_20-.mov | 0 → 0 | 46.51 → 54.26 | 70.28 → 74.59 |
| clip_35_own_video.webm | 0 → 0 | 19.66 → 11.97 | 55.37 → 51.09 |
| clip_37_own_video.webm | 0 → 0 | 50.00 → 47.20 | 72.22 → 70.67 |
| clip_40_own_video.webm | 2 → 1 | 0.00 → 0.00 | 0.00 → 0.00 |
| clip_44_own_video.webm | 0 → 0 | 44.94 → 48.99 | 69.41 → 71.66 |
| clip_46_own_video.webm | 1 → 1 | 0.41 → 0.00 | 2.05 → 0.00 |
| clip_47_own_video.webm | 1 → 0 | 1.06 → 4.26 | 5.30 → 21.30 |
| clip_48_own_video.webm | 1 → 1 | 2.15 → 0.54 | 10.75 → 2.70 |
| clip_52_own_video.webm | 1 → 1 | 0.41 → 0.00 | 2.05 → 0.00 |
| clip_54_own_video.webm | 0 → 0 | 10.80 → 13.20 | 50.44 → 51.78 |
| clip_55_own_video.webm | 0 → 0 | 50.00 → 46.47 | 72.22 → 70.26 |
| clip_60_bench_9_8.mp4 | 0 → 0 | 64.96 → 67.52 | 80.53 → 81.96 |
| clip_62_bench_anna.mp4 | 0 → 0 | 76.96 → 73.82 | 87.20 → 85.46 |
| clip_79_df_426504.mp4 | 0 → 0 | 98.52 → 96.30 | 99.18 → 97.94 |

Codes: 0 speaking, 1 not-speaking, 2 no-face. Clip 40 changes from no-face to
not-speaking; clip 47 changes from not-speaking to speaking. These are
regressions against the reference behavior, not measured human-label accuracy.

Removing lossy MP4 video/audio encoders changes model inputs; timestamps,
resampling and window boundaries may also contribute. This run does not
isolate the cause. Before considering adoption, compare frame counts/PTS,
audio sample windows, and intermediate features on the outliers, then test
video-only and audio-only changes independently. Do not silently widen the
gate or change decision thresholds to accommodate this experiment.

## Provenance and validation

- GPU service: `ml-asd-golden`, project `biometry-416410`, `us-central1`.
- Resources unchanged: one L4 GPU, 4 vCPU, 16Gi.
- Experimental revision: `ml-asd-golden-00004-hv6`.
- Image: `us-central1-docker.pkg.dev/biometry-416410/cloud-run-source-deploy/ml-asd-golden@sha256:a8cfb6a395b42a4d78332517c388f573a066c3a9301d206c8c9d957ab3eff889`.
- Cloud Build: `a01e3622-a5c0-4160-86a0-479ce03d1ce0`.
- Full run: 2026-10-21 16:58:40–17:02:18 UTC.
- Corpus: same approved 90 GCS objects from `baseline.json`; object contents
  are not generation-pinned.
- Full responses: `raw_media.json`; stage timings: `raw_media_timings.json`.
- Local validation: 16 tests passed (raw media plumbing, PCM conversion, golden
  comparator), plus Python compilation and whitespace checks. Local FFmpeg
  tests check media plumbing, not model inference; all inference ran on GPU.

Production was not modified. Staging traffic was restored to the validated
revision `ml-asd-golden-00003-hjh` (100%). The experimental revision remains
available for targeted investigation but receives no ordinary staging traffic.
