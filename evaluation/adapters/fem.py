import time,traceback
import numpy as np
from utils import resolve,pixel_hash,sha,digest,package_versions

_SOLVER=None
def initialize(cfg):
    global _SOLVER,_LIMIT
    from threadpoolctl import threadpool_limits
    from mechanics import FiniteSpecimenFEM
    if cfg['module_file']!='mechanics.py':raise ValueError('Use a different FEM adapter for a different solver implementation')
    _LIMIT=threadpool_limits(1);np.random.seed(cfg['preconditioner_seed'])
    _SOLVER=FiniteSpecimenFEM(**cfg['parameters'])
def solve(item):
    key,pixels=item;start=time.perf_counter()
    try:
        result=_SOLVER.solve(pixels)
        return dict(cache_key=key,input_pixel_sha256=pixel_hash(pixels),status='ok',**result)
    except Exception as e:
        return dict(cache_key=key,input_pixel_sha256=pixel_hash(pixels),status='failed',fem_valid=False,error=repr(e),traceback=traceback.format_exc(),solve_seconds=time.perf_counter()-start)
class OriginalFEM:
    initializer=staticmethod(initialize);solve=staticmethod(solve)
    @staticmethod
    def identity(cfg,root):return digest(dict(config=cfg,implementation=sha(resolve(cfg['module_file'],root)),adapter=sha(__file__),
        libraries=package_versions(['numpy','scipy','pyamg','threadpoolctl'])))
