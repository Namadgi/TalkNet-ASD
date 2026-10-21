"""Validated BM-494 MP4 + read_video path, kept separate from raw experiments.

Normalization arguments and decoded-audio conversion match serving source
6637385. The cropped-face XVID roundtrip lives downstream and is unchanged.
"""

import os
import subprocess
import time

from src.ml_models.audio import audio_to_pcm16


def validated_media_command(path, output='output.mp4'):
    cmd = ['ffmpeg', '-y']
    if path.lower().endswith('.webm'):
        cmd += ['-fflags', '+genpts']
    cmd += ['-i', path, '-vf', 'scale=-2:640', '-qscale:v', '2',
            '-async', '1', '-r', '25']
    if path.lower().endswith('.webm'):
        cmd += ['-max_muxing_queue_size', '1024']
    return cmd + ['-qscale:a', '0', '-ac', '1', '-threads', '10', '-ar', '16000',
                  '-loglevel', 'panic', output]


def read_validated_media(path, output='output.mp4'):
    # Lazy import: selecting raw must not invoke torchvision's media reader.
    from torchvision.io import read_video

    start = time.perf_counter()
    if os.path.isfile(output):
        os.remove(output)
    subprocess.run(validated_media_command(path, output), check=True, capture_output=True)
    normalized_at = time.perf_counter()
    video, aframes, info = read_video(os.path.abspath(output), end_pts=10, pts_unit='sec')
    if info.get('audio_fps') != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {info.get('audio_fps')}")
    return video.numpy(), audio_to_pcm16(aframes.detach().cpu().numpy()), {
        'normalize_unpack_seconds': normalized_at - start,
        'assemble_seconds': time.perf_counter() - normalized_at,
    }
