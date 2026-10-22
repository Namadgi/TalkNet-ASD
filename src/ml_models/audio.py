"""Audio conversion for the normalized 16 kHz mono media stream."""

import numpy as np


def audio_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert PyAV's normalized float or int16 mono samples to PCM16.

    Saturate before casting: +1.0 must become 32767, not wrap to -32768.
    Use nearest-integer rounding and saturate to the signed 16-bit range.
    """
    samples = np.asarray(samples)
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim != 1:
        raise ValueError("Expected normalized mono audio [1, samples]")
    if samples.dtype == np.int16:
        return samples
    if not np.issubdtype(samples.dtype, np.floating):
        raise ValueError(f"Unsupported decoded audio dtype: {samples.dtype}")
    if not np.isfinite(samples).all():
        raise ValueError("Decoded audio contains non-finite samples")
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int16)
