"""FFmpeg/PyAV media plumbing tests; no model inference or GPU benchmarking.

Run: python -m unittest tools.asd_eval.test_raw_media
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import av
import numpy as np

from src.ml_models.raw_media import read_raw_media


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class RawMediaTests(unittest.TestCase):
    def make_clip(self, directory, duration='0.4', audio_delay=None, audio=True):
        path = os.path.join(directory, 'source.mkv')
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
               '-i', 'color=c=red:size=64x128:rate=30:duration=' + duration]
        if audio:
            if audio_delay is not None:
                cmd += ['-itsoffset', str(audio_delay)]
            cmd += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=' + duration,
                    '-c:a', 'pcm_s16le']
        cmd += ['-c:v', 'ffv1', path]
        subprocess.run(cmd, check=True, capture_output=True)
        return path

    def test_normalized_shapes_and_rgb_order(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_clip(directory)
            video, audio, timings = read_raw_media(source)
            reference = os.path.join(directory, 'reference.mp4')
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error', '-i', source,
                '-vf', 'scale=-2:640', '-qscale:v', '2', '-async', '1',
                '-r', '25', '-qscale:a', '0', '-ac', '1', '-threads', '10',
                '-ar', '16000', reference,
            ], check=True, capture_output=True)
            with av.open(reference) as container:
                expected_frames = sum(1 for _ in container.decode(video=0))
        # Match the previous normalizer's frame count, including its tail
        # flush behavior (which can vary between FFmpeg versions).
        self.assertEqual(video.shape, (expected_frames, 640, 320, 3))
        self.assertEqual(video.dtype, np.uint8)
        self.assertEqual(audio.dtype, np.int16)
        self.assertAlmostEqual(len(audio) / 16000, 0.4, places=2)
        self.assertGreater(video[..., 0].mean(), 240)
        self.assertLess(video[..., 2].mean(), 10)
        self.assertGreater(timings['normalize_unpack_seconds'], 0)

    def test_ten_second_endpoint_is_inclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            video, audio, _ = read_raw_media(self.make_clip(directory, duration='11'))
        self.assertEqual(len(video), 251)
        self.assertEqual(len(audio), 160640)

    def test_delayed_audio_keeps_timestamp_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            _, audio, _ = read_raw_media(self.make_clip(directory, audio_delay=0.12))
        np.testing.assert_array_equal(audio[:1600], 0)
        self.assertGreater(np.abs(audio[2500:]).max(), 100)

    def test_missing_audio_and_bad_input_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (self.make_clip(directory, audio=False), os.path.join(directory, 'missing.mp4')):
                with self.subTest(path=path), self.assertRaises((ValueError, RuntimeError)):
                    read_raw_media(path)

    def test_timeout_stops_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_clip(directory)
            with self.assertRaises(TimeoutError):
                read_raw_media(path, timeout=0.00001)


if __name__ == '__main__':
    unittest.main()
