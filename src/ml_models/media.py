"""Explicit selection of independently maintained ASD preprocessing paths."""

import os


MEDIA_MODES = ('validated', 'raw', 'raw_video', 'raw_audio')


def get_media_mode():
    mode = os.getenv('ASD_MEDIA_MODE', 'validated')
    if mode not in MEDIA_MODES:
        raise ValueError(f'Unsupported ASD_MEDIA_MODE={mode!r}; expected one of {MEDIA_MODES}')
    return mode


def read_media(path):
    mode = get_media_mode()
    if mode == 'validated':
        from src.ml_models.validated_media import read_validated_media
        return read_validated_media(path)
    if mode == 'raw':
        from src.ml_models.raw_media import read_raw_media
        return read_raw_media(path)
    from src.ml_models.hybrid_media import read_hybrid_media
    return read_hybrid_media(path, mode)
