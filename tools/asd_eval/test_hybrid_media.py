"""Check the untouched stream against the original MP4 normalization path."""

import os
import subprocess
import tempfile
import unittest

import av
import numpy as np

from src.ml_models.hybrid_media import _retained_audio, _video_frames, read_hybrid_media
from src.ml_models.audio import audio_to_pcm16


class HybridMediaTests(unittest.TestCase):
    def reference(self, directory, source=None):
        if source is None:
            source = os.path.join(directory, 'source.mkv')
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                '-i', 'testsrc2=size=128x96:rate=30:duration=0.6', '-f', 'lavfi',
                '-i', 'sine=frequency=440:sample_rate=48000:duration=0.6',
                '-c:v', 'ffv1', '-c:a', 'pcm_s16le', source,
            ], check=True, capture_output=True)
        reference = os.path.join(directory, 'reference.mp4')
        cmd = ['ffmpeg', '-y', '-loglevel', 'error']
        if source.lower().endswith('.webm'):
            cmd += ['-fflags', '+genpts']
        cmd += ['-i', source, '-vf', 'scale=-2:640', '-qscale:v', '2',
                '-async', '1', '-r', '25', '-qscale:a', '0', '-ac', '1',
                '-threads', '10', '-ar', '16000', reference]
        subprocess.run(cmd, check=True, capture_output=True)
        with av.open(reference) as container:
            video = _video_frames(container)
        with av.open(reference) as container:
            audio = _retained_audio(container)
        return source, video, audio

    def test_video_only_change_preserves_audio_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            source, reference_video, reference_audio = self.reference(directory)
            video, audio, _ = read_hybrid_media(source, 'raw_video')
        np.testing.assert_array_equal(audio, reference_audio)
        self.assertEqual(video.shape, reference_video.shape)
        self.assertTrue(np.any(video != reference_video))

    def test_audio_only_change_preserves_video_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            source, reference_video, reference_audio = self.reference(directory)
            video, audio, _ = read_hybrid_media(source, 'raw_audio')
        np.testing.assert_array_equal(video, reference_video)
        # Old PyAV exposes AAC tail padding, newer PyAV discards it. Direct
        # PCM can therefore be up to one AAC frame shorter on this fixture.
        self.assertLessEqual(abs(len(audio) - len(reference_audio)), 1024)
        self.assertFalse(np.array_equal(audio, reference_audio))

    def test_invalid_mode_fails(self):
        with self.assertRaises(ValueError):
            read_hybrid_media('unused', 'invalid')

    def test_retained_audio_matches_reference_seek(self):
        with tempfile.TemporaryDirectory() as directory:
            self.reference(directory)
            reference = os.path.join(directory, 'reference.mp4')
            with av.open(reference) as container:
                container.seek(0, any_frame=False, backward=True, stream=container.streams.audio[0])
                expected = audio_to_pcm16(np.concatenate([
                    frame.to_ndarray() for frame in container.decode(audio=0)
                    if 0 <= frame.pts * frame.time_base <= 10
                ], axis=1))
            with av.open(reference) as container:
                actual = _retained_audio(container)
        np.testing.assert_array_equal(actual, expected)

    def test_timeout_stops_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = self.reference(directory)
            with self.assertRaises(TimeoutError):
                read_hybrid_media(source, 'raw_audio', timeout=0.00001)


if __name__ == '__main__':
    unittest.main()
