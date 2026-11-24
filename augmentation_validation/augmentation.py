"""Exact square D4 augmentation; rotate CCW first, then flip left/right.

Only Ex/Ey mapping is supplied. Do not reuse it for arbitrary tensor labels.
Input is a 2D binary material map, 0 hard and 1 soft, before [-1,1] conversion.
"""
import numpy as np


def transform(image, quarter_turns, horizontal_flip):
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not np.isin(arr, (0, 1)).all():
        raise ValueError('Expected a square 2D binary material map.')
    if quarter_turns not in (0, 1, 2, 3):
        raise ValueError('quarter_turns must be 0,1,2,3')
    result = np.rot90(arr, k=quarter_turns, axes=(0, 1))
    if horizontal_flip:
        result = np.flip(result, axis=1)
    # copy also prevents an identity result from aliasing a writable input.
    return np.array(result, dtype=arr.dtype, order='C', copy=True)


def map_ex_ey(ex, ey, quarter_turns):
    if quarter_turns not in (0, 1, 2, 3):
        raise ValueError('quarter_turns must be 0,1,2,3')
    return (ey, ex) if quarter_turns % 2 else (ex, ey)


def random_augmentation(image, ex, ey, rng):
    """Caller supplies its seeded RNG. Draw one uniform angle and independent flip."""
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    return transform(image, k, flip), map_ex_ey(ex, ey, k), dict(quarter_turns=k, horizontal_flip=flip)
