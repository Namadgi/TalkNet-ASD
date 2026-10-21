# BM-494: video-only and audio-only alternatives, 2026-10-21

Neither alternative satisfies the existing golden gate. No merge, push or PR.
The validated `bm-494-decode-once` branch and production remain unchanged.

## Results

| Alternative | GPU clips evaluated | Exact decision codes | Numeric gate | Mean HTTP latency |
|---|---:|---|---|---|
| Validated Stage 1 | 90 | Reference | Reference | 4.664s |
| Earlier all-raw experiment | 90 | 88/90 | 12 additional flags | 2.413s |
| Raw video, reference AAC audio | 2 targeted clips | 0/2 | Rejected on decisions | Not benchmarked |
| Reference H.264 video, raw PCM audio | 90 | 90/90 | 11 flags vs Stage 1 | 4.434s |

Audio-only median HTTP latency: 4.100s, versus 4.511s for Stage 1. Its observed
mean reduction is 4.9%, not a large speedup. Comparisons are single sequential
runs at different times, not interleaved throughput benchmarks. Some latency
variation occurred; do not treat the percentage as a guaranteed benefit.
The warm-up/probe requests are excluded. Video-only's two cold/warm probe
latencies are not a performance comparison and must not be averaged against
the 90-clip baseline.

The earlier raw decision changes were independently reproduced before these
tests (`raw_decision_repro.json`). The two video-only probe responses also
reproduce them:

| Clip | Reference | Video-only | Audio-only |
|---|---|---|---|
| clip_40_own_video.webm | code 2, no face | code 1, not speaking | code 2, no face |
| clip_47_own_video.webm | code 1, result 1.06, score 5.30 | code 0, result 4.26, score 21.30 | code 1, result 1.60, score 8.00 |

Thus the video replacement alone is sufficient to trigger both decision
changes in the targeted tests. The audio-only change preserves all three-way
codes on the full set, including all nine no-face outcomes. This does not
establish accuracy against human labels or guarantee unseen clips will agree.

Audio-only has 79/90 complete gate passes, 11 numeric flags, no request errors,
and 43 exactly unchanged responses compared with validated Stage 1. Its
maximum result difference is 13.18 points and maximum score difference 7.32.
Compared with the original baseline, it has 80 complete passes and 10 numeric
flags (the decision agreement remains 90/90).

## Audio-only numeric outliers versus Stage 1

| Clip | Result old → candidate | Score old → candidate |
|---|---|---|
| clip_20_own_IMG_3866.mov | 85.71 → 88.89 | 92.06 → 93.83 |
| clip_21_own_Movie_on_20-.mov | 46.51 → 33.33 | 70.28 → 62.96 |
| clip_35_own_video.webm | 19.66 → 17.09 | 55.37 → 53.94 |
| clip_44_own_video.webm | 44.94 → 48.58 | 69.41 → 71.43 |
| clip_46_own_video.webm | 0.41 → 0.00 | 2.05 → 0.00 |
| clip_47_own_video.webm | 1.06 → 1.60 | 5.30 → 8.00 |
| clip_48_own_video.webm | 2.15 → 2.69 | 10.75 → 13.45 |
| clip_52_own_video.webm | 0.41 → 0.00 | 2.05 → 0.00 |
| clip_55_own_video.webm | 50.00 → 47.06 | 72.22 → 70.59 |
| clip_60_bench_9_8.mp4 | 64.96 → 68.38 | 80.53 → 82.43 |
| clip_79_df_426504.mp4 | 98.52 → 96.30 | 99.18 → 97.94 |

## Implementation and controlled checks

Both alternatives use one FFmpeg process to read the original input, emitting
one raw stream through a NUT pipe and writing the retained compressed stream
to a request-local temporary file. Video-only keeps AAC in M4A; audio-only
keeps H.264 in MP4. The cropped-face AVI roundtrip and model are unchanged.

Unlike the earlier all-raw trial, these variants normalize the full source to
preserve the reference encoder's lookahead/tail behavior. Raw audio concatenates
selected samples rather than inserting new leading silence for positive PTS.
Its end window allows the old AAC packet interval through 10.048s, but raw PCM
does not reproduce AAC tail padding or codec artifacts on short clips. Those
differences are part of the audio replacement, not changes to reference video.

A real control mismatch was caught before GPU evaluation. PyAV 12.3 changes
AAC startup samples when decoding from a fresh container versus seeking to
zero first. The reference `read_video` seeks first; the initial custom reader
did not. A synthetic regression test failed on 576/10240 samples, then passed
after reproducing the seek. The newer Mac PyAV did not expose the same behavior.
This was a reader-state mismatch, not evidence that the full encoder context
had to be retained. The superseded full-reference control modes were removed.

The successful build verifies against actual `torchvision.read_video` in the
immutable baseline image (PyAV 12.3), using short and >10s fixtures:

- Video-only retained audio: sample-for-sample identical on both fixtures.
- Audio-only retained video: pixel-for-pixel identical on both fixtures.
- Frame counts match: 17/17 and 251/251.
- Raw audio sample counts: 9600 vs reference 10240 on the short fixture, and
  160768/160768 on the long fixture, exposing the AAC-padding distinction.

These checks validate the fixtures, not every intermediate array in all 90
clips. The full GPU response gate is the final behavioral test. Twenty-one
local tests passed using PyAV 12.3; the hybrid tests also pass with PyAV 19.
Local tests are codec plumbing, not CPU-only model evaluation.

## Stage timings: mean seconds

| Stage | Stage 1 | Audio-only |
|---|---:|---:|
| GCS download | 0.218 | 0.170 |
| Normalize/pipe processing, excluding download | 2.190 | 2.137 |
| Read/assemble retained media | 0.808 | 0.692 |
| Face detection | 0.362 | 0.360 |
| Face cropping, conditional on a track | 0.294 | 0.279 |
| Feature preparation, conditional on a track | 0.072 | 0.067 |
| TalkNet inference, conditional on a track | 0.471 | 0.470 |

The final three rows cover 81 tracked clips; other rows cover 90. The second
and third rows have different internal work across implementations; their
combined time is the comparable preprocessing total (2.999s → 2.828s).

## Provenance and cleanup

- Experimental branch: `bm-494-raw-preprocessing`; serving source `d8f8fa8`.
- Fixed dependency/codec/weight baseline image is unchanged.
- Verified build: `23399c09-b69e-4d0d-85c6-f803fc02a3c9`.
- Image: `us-central1-docker.pkg.dev/biometry-416410/cloud-run-source-deploy/ml-asd-golden@sha256:b6b46343253004799e2d74ecec928850c821572c2c145868e619af8f60482bfc`.
- Audio-only revision: `ml-asd-golden-00007-mux`, `ASD_MEDIA_MODE=raw_audio`.
- Video-only revision: `ml-asd-golden-00008-lug`, `ASD_MEDIA_MODE=raw_video`.
- Full audio run: 2026-10-21 17:37:17–17:43:56 UTC, same 90 GCS object names.
- Responses: `hybrid_audio.json`; stage timings: `hybrid_audio_timings.json`;
  targeted probes: `hybrid_audio_probe.json` and `hybrid_video_probe.json`.
- Temporary test URLs/tags were removed after evaluation. Ordinary staging
  traffic remains 100% on `ml-asd-golden-00003-hjh`; production was not touched.
- The deploy helper now explicitly sets or clears `ASD_MEDIA_MODE` so later
  deployments do not silently inherit an earlier experiment's setting.

Recommendation: keep the validated Stage 1 implementation. Video-only fails
the decision gate; audio-only trades numeric stability for only a modest
observed latency improvement. A future experiment can test a faster MP4
encoder preset while retaining AAC, independently of these raw-media changes.
