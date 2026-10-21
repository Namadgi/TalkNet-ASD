"""Integration plumbing only; GPU inference results live in results/.

Readers are mocked so these tests do not load CUDA models or torchvision.
"""
import os
import sys
import types
import unittest
from unittest.mock import patch, Mock

import numpy as np

from src.ml_models.media import get_media_mode, read_media
from src.ml_models.validated_media import read_validated_media, validated_media_command


class MediaSelectionTests(unittest.TestCase):
    def test_validated_is_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_media_mode(), 'validated')

    def test_selects_each_reader_without_fallback(self):
        for mode in ('validated', 'raw', 'raw_video', 'raw_audio'):
            name = 'validated_media' if mode == 'validated' else (
                'raw_media' if mode == 'raw' else 'hybrid_media')
            func = 'read_' + name
            reader = Mock(return_value=('video', 'audio', {}))
            module = types.ModuleType('src.ml_models.' + name)
            setattr(module, func, reader)
            with self.subTest(mode=mode), patch.dict(os.environ, ASD_MEDIA_MODE=mode), \
                    patch.dict(sys.modules, {'src.ml_models.' + name: module}):
                self.assertEqual(read_media('input.webm'), ('video', 'audio', {}))
                expected = ('input.webm', mode) if name == 'hybrid_media' else ('input.webm',)
                reader.assert_called_once_with(*expected)

    def test_invalid_modes_are_rejected(self):
        for mode in ('', 'RAW', 'typo'):
            with patch.dict(os.environ, ASD_MEDIA_MODE=mode), self.assertRaises(ValueError):
                read_media('input.mp4')

    def test_reader_failure_does_not_silently_switch_approaches(self):
        module = types.ModuleType('src.ml_models.raw_media')
        module.read_raw_media = Mock(side_effect=RuntimeError('codec failed'))
        with patch.dict(os.environ, ASD_MEDIA_MODE='raw'), \
                patch.dict(sys.modules, {'src.ml_models.raw_media': module}), \
                self.assertRaisesRegex(RuntimeError, 'codec failed'):
            read_media('input.mp4')

    def test_validated_command_preserves_reference_arguments(self):
        expected = ['ffmpeg', '-y', '-i', 'input.mp4', '-vf', 'scale=-2:640',
                    '-qscale:v', '2', '-async', '1', '-r', '25', '-qscale:a', '0',
                    '-ac', '1', '-threads', '10', '-ar', '16000',
                    '-loglevel', 'panic', 'output.mp4']
        self.assertEqual(validated_media_command('input.mp4'), expected)
        webm = validated_media_command('input.WEBM')
        self.assertEqual(webm[2:4], ['-fflags', '+genpts'])
        self.assertEqual(webm[webm.index('-max_muxing_queue_size') + 1], '1024')
        self.assertNotIn('-t', webm)  # The reference normalizes the full source.

    def decoded(self, rate=16000):
        video = Mock()
        video.numpy.return_value = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        audio = Mock()
        audio.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [[-1., 0., 1.]], dtype=np.float32)
        module = types.ModuleType('torchvision.io')
        module.read_video = Mock(return_value=(video, audio, {'audio_fps': rate}))
        return module

    def test_validated_reader_preserves_window_audio_and_cleanup(self):
        module = self.decoded()
        with patch.dict(sys.modules, {'torchvision.io': module}), \
                patch('src.ml_models.validated_media.subprocess.run') as run, \
                patch('src.ml_models.validated_media.os.path.isfile', return_value=True), \
                patch('src.ml_models.validated_media.os.remove') as remove:
            video, audio, timings = read_validated_media('input.mp4')
            run.assert_called_once_with(validated_media_command('input.mp4'),
                                        check=True, capture_output=True)
            remove.assert_called_once_with('output.mp4')
            module.read_video.assert_called_once_with(os.path.abspath('output.mp4'),
                                                     end_pts=10, pts_unit='sec')
            np.testing.assert_array_equal(audio, [-32768, 0, 32767])
            self.assertEqual(video.shape, (2, 4, 4, 3))
            self.assertEqual(set(timings), {'normalize_unpack_seconds', 'assemble_seconds'})

    def test_validated_reader_rejects_wrong_sample_rate(self):
        with patch.dict(sys.modules, {'torchvision.io': self.decoded(48000)}), \
                patch('src.ml_models.validated_media.subprocess.run'), \
                patch('src.ml_models.validated_media.os.path.isfile', return_value=False), \
                self.assertRaisesRegex(ValueError, '16000 Hz'):
            read_validated_media('input.mp4')


if __name__ == '__main__':
    unittest.main()
