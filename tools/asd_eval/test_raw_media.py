"""FFmpeg/PyAV media plumbing tests; no model inference or GPU benchmarking.

Run: python -m unittest tools.asd_eval.test_raw_media
"""

import os
import shutil
import subprocess
import tempfile
import unittest

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
            video, audio, timings = read_raw_media(self.make_clip(directory))
        # FFmpeg's legacy -r conversion flushes two tail frames for this
        # 30fps fixture, just like the previous MP4 normalization command.
        self.assertEqual(video.shape, (12, 640, 320, 3))
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


if __name__ == '__main__':
    unittest.main()
