"""Run from the repository root: python3 -m unittest tools.asd_eval.test_audio."""

import unittest

import numpy as np

from src.ml_models.audio import audio_to_pcm16


class AudioConversionTests(unittest.TestCase):
    def test_peaks_saturate_instead_of_wrapping(self):
        samples = np.array([[-1.2, -1, 0, 1, 1.2]], dtype=np.float32)
        np.testing.assert_array_equal(
            audio_to_pcm16(samples), [-32768, -32768, 0, 32767, 32767]
        )

    def test_rounding_to_nearest_sample(self):
        samples = np.array([0.6, -0.6, 1.6, -1.6], dtype=np.float32) / 32768
        np.testing.assert_array_equal(audio_to_pcm16(samples), [1, -1, 2, -2])

    def test_existing_pcm_is_preserved(self):
        samples = np.array([[-32768, -1, 0, 32767]], dtype=np.int16)
        np.testing.assert_array_equal(audio_to_pcm16(samples), samples[0])

    def test_invalid_audio_is_rejected(self):
        for samples in (np.zeros((2, 8)), np.array([np.nan]), np.array([1], dtype=np.int32)):
            with self.subTest(shape=samples.shape, dtype=samples.dtype):
                with self.assertRaises(ValueError):
                    audio_to_pcm16(samples)


if __name__ == "__main__":
    unittest.main()
