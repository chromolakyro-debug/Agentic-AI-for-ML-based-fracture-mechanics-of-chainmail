"""Core invariants; no synthetic data is used as scientific evaluation results."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from utils import HERE,ROOT,read,save,digest,to_sa,from_sa,sha
from cache import get,put,valid_array
from retrieval import nearest,target_cache_key
from targets import make_targets
from pipeline import summarize,Run,atomic_npz

class Invariants(unittest.TestCase):
    def setUp(self):self.distance=dict(scales=[1.,1.],allow_axis_swap=True)
    def test_swap_direction_and_unique_source(self):
        p=np.array([[2.,8.],[8.,2.],[3.,7.],[7.,3.]])
        result=nearest([8.,2.],p,['same','same','other','third'],np.arange(4),self.distance,3)
        self.assertEqual([r['dataset_index'] for r in result],[0,2,3]);self.assertTrue(result[0]['axis_swapped']);self.assertEqual(result[0]['distance'],0)
        np.testing.assert_array_equal([result[0]['Ex_matched'],result[0]['Ey_matched']],[8.,2.])
    def test_training_scope_and_tie(self):
        result=nearest([2.,8.],np.array([[2.,8.],[3.,7.],[4.,6.]]),['val','train','test'],np.array([1]),self.distance,1)
        self.assertEqual(result[0]['dataset_index'],1);self.assertFalse(result[0]['axis_swapped'])
    def test_fixed_anisotropic_scales_both_directions(self):
        c=dict(scales=[2.,.5],allow_axis_swap=True);p=np.array([[3.,7.],[6.,2.]])
        actual=nearest([8.,2.],p,['a','b'],np.arange(2),c,2)
        expected=[min(np.linalg.norm((np.array([8,2])-row)/c['scales']),np.linalg.norm((np.array([8,2])-row[::-1])/c['scales'])) for row in p]
        for r in actual:self.assertAlmostEqual(r['distance'],expected[r['dataset_index']])
    def test_cache_invalidates_and_corruption_detected(self):
        base=target_cache_key('train image+label+split hash',[{'s':1}],self.distance,10,'code')
        for x in [target_cache_key('changed image',[{'s':1}],self.distance,10,'code'),target_cache_key('train image+label+split hash',[{'s':2}],self.distance,10,'code'),target_cache_key('train image+label+split hash',[{'s':1}],dict(scales=[2,1]),10,'code'),target_cache_key('train image+label+split hash',[{'s':1}],self.distance,10,'new code')]:self.assertNotEqual(base,x)
        (HERE/'cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'cache') as td:
            p=Path(td)/'test.json';put(p,base,{'status':'failed','error':'recorded'});self.assertEqual(get(p,base)['status'],'failed');self.assertIsNone(get(p,'wrong'))
            d=read(p);d['payload']['status']='ok';save(p,d);self.assertIsNone(get(p,base))
    def test_formula_and_exact_cell_midpoints(self):
        e=np.array([[2.,8.],[7.,3.]]);sa=to_sa(e[:,0],e[:,1],1);np.testing.assert_allclose(from_sa(sa[:,0],sa[:,1],1),e,rtol=1e-14)
        np.testing.assert_allclose(to_sa(e[:,1],e[:,0],1),sa*np.array([1,-1]))
        g=read(HERE/'configs/grid.json');ts=make_targets([[r,c] for r in range(1,5) for c in range(1,7)],g)
        self.assertEqual(len(ts),24);self.assertEqual(ts[0]['a_target'],np.mean(g['a_edges'][:2]));self.assertEqual(ts[-1]['grid_col'],6)
        with self.assertRaises(ValueError):make_targets([[1,1]]*24,g)
    def test_image_content_array_corruption_and_fem_config_invalidation(self):
        from adapters.fem import OriginalFEM
        import copy
        cfg=read(HERE/'configs/old_20k.json')['fem'];altered=copy.deepcopy(cfg);altered['parameters']['tolerance']=1e-10
        identity=OriginalFEM.identity(cfg,ROOT);other=OriginalFEM.identity(altered,ROOT);self.assertNotEqual(identity,other)
        run=Run.__new__(Run);run.fem_identity=identity
        a=np.zeros((4,4),dtype=np.uint8);b=a.copy();b[0,0]=1
        key=run.fem_key(a);self.assertNotEqual(key,run.fem_key(b));run.fem_identity=other;self.assertNotEqual(key,run.fem_key(a))
        with tempfile.TemporaryDirectory(dir=HERE/'cache') as td:
            p=Path(td)/'pixels.npz';meta=p.with_suffix('.json');atomic_npz(p,binary=a)
            put(meta,'frozen',dict(file_sha256=sha(p)));self.assertIsNotNone(valid_array(p,meta,'frozen'))
            atomic_npz(p,binary=b);self.assertIsNone(valid_array(p,meta,'frozen'))
    def test_sample_variance_and_failures(self):
        targets=[dict(target_id='A',grid_row=1,grid_col=2),dict(target_id='B',grid_row=2,grid_col=2)]
        def good(t,v):return dict(target_id=t,status='ok',d_mech=v,hit_Ex_Ey=True,hit_s_a=True,delta_Ex=v,delta_Ey=v,delta_s=v,delta_a=v)
        rr=[good('A',1),good('A',3),dict(target_id='A',status='failed'),good('B',2)]
        s=summarize(rr,targets,1);self.assertEqual(s[0]['mean_d_mech'],2);self.assertEqual(s[0]['var_d_mech'],2);self.assertEqual(s[0]['joint_s_a_hit_rate'],2/3);self.assertIsNone(s[1]['var_d_mech'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Invariants))
    save(HERE/'validation/core_tests.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
