"""CFM v0 shared paths and reproducible, immutable experiment configuration."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'cat_binary_dataset'
OUT = Path(os.environ.get('CFM_V0_RUN_DIR', str(HERE / 'run'))).resolve()
SEED = 20261114
S_TOL, A_TOL = float(np.log(1.05)), float(np.log(1.10))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def to_sa(ex, ey, esoft=1.):
    ex, ey = np.asarray(ex), np.asarray(ey)
    if np.any(ex <= 0) or np.any(ey <= 0) or esoft <= 0:
        raise ValueError('Young moduli must be positive.')
    return np.stack([.5 * (np.log(ex / esoft) + np.log(ey / esoft)), np.log(ex / ey)], axis=-1)


def from_sa(sa, esoft=1.):
    sa = np.asarray(sa)
    return esoft * np.stack([np.exp(sa[..., 0] + sa[..., 1] / 2),
                             np.exp(sa[..., 0] - sa[..., 1] / 2)], axis=-1)
