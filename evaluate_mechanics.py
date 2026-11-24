"""Resumable full-resolution specimen labelling; working files live outside release."""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

_SOLVER = None
_THREAD_LIMIT = None
_PROTOCOL_FINGERPRINT = None


def mechanics_fingerprint():
    from mechanics import FiniteSpecimenFEM, q4_stiffness
    identity={"shape":[224,224],"soft_modulus":1.,"hard_modulus":10.,"poisson":.3,
              "width":40.,"height":40.,"thickness":2.,"strain":.001,
              "solver":"amg","tolerance":1e-9,
              "implementation":inspect.getsource(FiniteSpecimenFEM)+inspect.getsource(q4_stiffness)}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode("utf-8")).hexdigest()


def initialize_worker():
    global _SOLVER, _THREAD_LIMIT, _PROTOCOL_FINGERPRINT
    from threadpoolctl import threadpool_limits
    from mechanics import FiniteSpecimenFEM
    _THREAD_LIMIT = threadpool_limits(1)
    _SOLVER = FiniteSpecimenFEM(shape=(224,224),soft_modulus=1.,hard_modulus=10.,poisson=.3,
                              width=40.,height=40.,thickness=2.,strain=.001,solver="amg",tolerance=1e-9)
    _PROTOCOL_FINGERPRINT=mechanics_fingerprint()


def solve_item(item):
    name, pixels = item
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    result = _SOLVER.solve(pixels)
    return dict(key=name, sha256_pixels=digest, protocol_fingerprint=_PROTOCOL_FINGERPRINT, **result)


def label_items(items, destination, workers=8):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol=mechanics_fingerprint()
    cached = {}
    if destination.exists():
        for line in destination.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("protocol_fingerprint")==protocol:
                cached[(row["key"], row["sha256_pixels"])] = row
    pending = [(name, arr) for name, arr in items
               if (name, hashlib.sha256(arr.tobytes()).hexdigest()) not in cached]
    print(f"FEM: {len(items)-len(pending)} cached, {len(pending)} pending, {workers} workers", flush=True)
    start = time.perf_counter()
    with destination.open("a", encoding="utf-8") as stream:
        if pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
                futures = [executor.submit(solve_item, item) for item in pending]
                for index, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    cached[(row["key"], row["sha256_pixels"])] = row
                    if index % 25 == 0 or index == len(pending):
                        print(f"FEM: {index}/{len(pending)} complete, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    return [cached[(name, hashlib.sha256(arr.tobytes()).hexdigest())] for name, arr in items]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    items = []
    if args.images:
        for path in sorted(args.images.glob("cat_*.png")):
            with Image.open(path) as image:
                items.append((path.name, np.asarray(image, dtype=np.uint8)))
    elif args.cache:
        for path in sorted(args.cache.glob("batch_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                rows = json.loads(archive["rows"].item())
                for row, pixels in zip(rows, archive["pixels"]):
                    items.append((str(row["spec"]["candidate"]), pixels))
    else:
        raise ValueError("Provide --images or --cache")
    if args.limit:
        items = items[:args.limit]
    labels = label_items(items, args.destination, args.workers)
    values = np.array([[row["Ex_app"], row["Ey_app"]] for row in labels])
    print(json.dumps(dict(n=len(labels), minimum=values.min(axis=0).tolist(),
                         maximum=values.max(axis=0).tolist(), median=np.median(values, axis=0).tolist())))


if __name__ == "__main__":
    main()
