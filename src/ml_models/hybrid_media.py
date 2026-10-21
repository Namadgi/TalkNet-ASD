"""Controlled BM-494 experiments: bypass only one normalized media encoder.

Keep the other stream's original MP4/M4A encoding and read semantics. Using
separate outputs avoids shifting video timestamps to accommodate AAC priming
in the NUT pipe. One FFmpeg process reads the original input once.
"""

import os
import subprocess
import tempfile
import threading
import time

import av
import numpy as np

from src.ml_models.audio import audio_to_pcm16
from src.ml_models.raw_media import END_SECONDS, FPS, SAMPLE_RATE


def hybrid_command(path, retained_path, mode):
    if mode not in ('raw_video', 'raw_audio'):
        raise ValueError('Unknown hybrid media mode')
    cmd = ['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error']
    if path.lower().endswith('.webm'):
        cmd += ['-fflags', '+genpts']
    cmd += ['-i', path]
    video_options = ['-vf', 'scale=-2:640', '-r', str(FPS), '-threads', '10']
    audio_options = ['-async', '1', '-ac', '1', '-ar', str(SAMPLE_RATE), '-threads', '10']
    if mode == 'raw_video':
        cmd += ['-map', '0:v:0'] + video_options + [
            '-vsync', 'cfr', '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p', '-f', 'nut', 'pipe:1',
            '-map', '0:a:0',
        ] + audio_options + ['-c:a', 'aac', '-qscale:a', '0', retained_path]
    else:
        cmd += ['-map', '0:a:0'] + audio_options + [
            '-c:a', 'pcm_s16le', '-f', 'nut', 'pipe:1', '-map', '0:v:0',
        ] + video_options + ['-c:v', 'libx264', '-qscale:v', '2', retained_path]
    # No -t: keep the old encoder's full source lookahead and tail behavior.
    return cmd


def _video_frames(container, drain=False):
    frames = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            raise ValueError('Normalized video timestamp missing')
        position = frame.pts * frame.time_base
        if position > END_SECONDS:
            if not drain:
                break
        elif position >= 0:
            frames.append(frame.to_ndarray(format='rgb24'))
    if not frames or len(frames) > END_SECONDS * FPS + 1:
        raise ValueError('Unexpected normalized video frame count')
    return np.stack(frames)


def _retained_audio(container):
    # Match torchvision.read_video: select AAC frames with PTS in [0, 10]
    # inclusive and concatenate their samples, including final AAC padding.
    # torchvision seeks before decoding. With PyAV 12.3, skipping this seek
    # changes AAC startup samples even for an otherwise identical bitstream.
    container.seek(0, any_frame=False, backward=True, stream=container.streams.audio[0])
    frames = []
    for frame in container.decode(audio=0):
        if frame.pts is None or frame.sample_rate != SAMPLE_RATE:
            raise ValueError('Invalid retained audio timestamp/sample rate')
        position = frame.pts * frame.time_base
        if position > END_SECONDS:
            break
        if position >= 0:
            frames.append(frame.to_ndarray())
    if not frames:
        raise ValueError('No retained audio samples')
    return audio_to_pcm16(np.concatenate(frames, axis=1))


def _raw_audio(container):
    # Preserve the reference AAC packet window length without encoding AAC:
    # frames starting at/before 10s include samples through 10.048s (16kHz).
    limit = (END_SECONDS * SAMPLE_RATE // 1024 + 1) * 1024
    parts = []
    for frame in container.decode(audio=0):
        if frame.pts is None or frame.sample_rate != SAMPLE_RATE:
            raise ValueError('Invalid raw audio timestamp/sample rate')
        offset = round(frame.pts * frame.time_base * SAMPLE_RATE)
        samples = audio_to_pcm16(frame.to_ndarray())
        if offset < 0:
            samples, offset = samples[-offset:], 0
        if offset < limit and len(samples):
            parts.append(samples[:limit - offset].copy())
    if not parts:
        raise ValueError('No raw audio samples')
    # The reference reader concatenates selected frames, ignoring leading PTS
    # offsets/gaps. Do not introduce new leading silence as the all-raw trial
    # did. Short-clip AAC discard padding still depends on decoder version.
    return np.concatenate(parts)


def read_hybrid_media(path, mode, timeout=120):
    start = time.perf_counter()
    video_raw = mode.endswith('_video')
    with tempfile.TemporaryDirectory(prefix='asd-hybrid-') as directory:
        retained_path = os.path.join(directory, 'retained.m4a' if mode == 'raw_video' else 'retained.mp4')
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(hybrid_command(path, retained_path, mode),
                                       stdout=subprocess.PIPE, stderr=errors)
            expired = threading.Event()

            def expire():
                expired.set()
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

            timer = threading.Timer(timeout, expire)
            timer.daemon = True
            timer.start()
            try:
                with av.open(process.stdout, mode='r', format='nut') as container:
                    raw = _video_frames(container, drain=True) if video_raw else _raw_audio(container)
                result = process.wait()
                if expired.is_set():
                    raise TimeoutError('Hybrid normalization timed out')
                if result:
                    raise RuntimeError('FFmpeg hybrid normalization failed')
            except Exception as exc:
                if process.poll() is None:
                    process.kill()
                if expired.is_set():
                    raise TimeoutError('Hybrid normalization timed out') from exc
                errors.seek(0)
                raise RuntimeError('Hybrid normalization failed: ' + errors.read(2000).decode(errors='replace')) from exc
            finally:
                timer.cancel()
                process.stdout.close()
                process.wait()
        normalized_at = time.perf_counter()
        with av.open(retained_path) as container:
            retained = _retained_audio(container) if video_raw else _video_frames(container)
        video, audio = (raw, retained) if video_raw else (retained, raw)
        return video, audio, {
            'normalize_unpack_seconds': normalized_at - start,
            'assemble_seconds': time.perf_counter() - normalized_at,
        }
