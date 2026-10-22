"""Decode source media once into normalized RGB video and PCM16 audio.

FFmpeg writes uncompressed YUV video and PCM16 audio into a timestamped NUT
pipe. PyAV only unpacks raw frames (and converts YUV to RGB), not a second
compressed video/audio stream. The face-crop AVI roundtrip is unchanged.
"""

import subprocess
import tempfile
import threading
import time
from typing import Tuple

import av
import numpy as np

from src.ml_models.audio import audio_to_pcm16


FPS = 25
SAMPLE_RATE = 16000
END_SECONDS = 10
# read_video(end_pts=10) includes the frame at exactly 10.0 seconds. Allow
# FFmpeg one extra frame; filter by PTS below, not by a rounded duration.
OUTPUT_SECONDS = END_SECONDS + 2 / FPS


def raw_media_command(path: str):
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error']
    if path.lower().endswith('.webm'):
        cmd += ['-fflags', '+genpts']
    cmd += [
        '-i', path, '-map', '0:v:0', '-map', '0:a:0',
        '-vf', 'scale=-2:640', '-r', str(FPS), '-vsync', 'cfr',
        '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p',
        '-async', '1', '-ac', '1', '-ar', str(SAMPLE_RATE),
        '-c:a', 'pcm_s16le', '-threads', '10',
        '-t', str(OUTPUT_SECONDS), '-f', 'nut', 'pipe:1',
    ]
    return cmd


def read_raw_media(path: str, timeout: float = 120) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return RGB THWC uint8 video, mono PCM16 audio, and stage timings.

    Drain a single interleaved pipe, so neither audio nor video can deadlock
    behind an unread second pipe. Preserve audio timestamp gaps with silence;
    never independently reset the two streams' origins. No MP4 is produced.
    """
    frames = []
    audio_parts = []
    start = time.perf_counter()
    expired = threading.Event()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(raw_media_command(path), stdout=subprocess.PIPE, stderr=errors)

        def expire():
            expired.set()
            process.kill()

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            with av.open(process.stdout, mode='r', format='nut') as container:
                if not container.streams.video or not container.streams.audio:
                    raise ValueError('Expected video and audio streams')
                for packet in container.demux():
                    for frame in packet.decode():
                        if frame.pts is None:
                            raise ValueError('Normalized frame has no timestamp')
                        position = frame.pts * frame.time_base
                        if packet.stream.type == 'video':
                            if 0 <= position <= END_SECONDS:
                                if len(frames) >= END_SECONDS * FPS + 1:
                                    raise ValueError('Unexpected normalized video frame count')
                                frames.append(frame.to_ndarray(format='rgb24'))
                        elif packet.stream.type == 'audio':
                            if frame.sample_rate != SAMPLE_RATE:
                                raise ValueError('Expected 16000 Hz normalized audio')
                            offset = round(position * SAMPLE_RATE)
                            samples = audio_to_pcm16(frame.to_ndarray())
                            if offset < 0:
                                samples = samples[-offset:]
                                offset = 0
                            # Include one video's frame of audio beyond 10s,
                            # because the last included video frame spans it.
                            stop = (END_SECONDS * FPS + 1) * (SAMPLE_RATE // FPS)
                            if offset < stop and len(samples):
                                audio_parts.append((offset, samples[:stop - offset].copy()))
            returncode = process.wait()
            if expired.is_set():
                raise TimeoutError('Raw media normalization timed out')
            if returncode:
                errors.seek(0)
                raise RuntimeError('FFmpeg raw normalization failed: ' + errors.read(2000).decode(errors='replace'))
        except Exception as exc:
            if process.poll() is None:
                process.kill()
            if expired.is_set():
                raise TimeoutError('Raw media normalization timed out') from exc
            errors.seek(0)
            detail = errors.read(2000).decode(errors='replace')
            if detail:
                raise RuntimeError('Raw media read failed: ' + detail) from exc
            raise
        finally:
            timer.cancel()
            process.stdout.close()
            process.wait()

    unpacked_at = time.perf_counter()
    if not frames or not audio_parts:
        raise ValueError('Normalized media contains no video/audio samples')
    # Copy once into a contiguous video array and release packet/frame buffers.
    video = np.stack(frames)
    audio = np.zeros(max(offset + len(samples) for offset, samples in audio_parts), dtype=np.int16)
    for offset, samples in audio_parts:
        audio[offset:offset + len(samples)] = samples
    return video, audio, {
        'normalize_unpack_seconds': unpacked_at - start,
        'assemble_seconds': time.perf_counter() - unpacked_at,
    }
