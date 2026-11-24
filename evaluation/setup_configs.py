"""Freeze project-specific inputs once; does not invent the missing selected cells."""
import copy
import numpy as np
from utils import ROOT,HERE,read,save,sha,from_sa

def main():
    out=HERE/'configs';out.mkdir(exist_ok=True)
    original=ROOT/'analysis/curated_augmented_distribution/grid_v1_equal_width/s_a/counts.json'
    grid=read(original)
    save(out/'grid.json',dict(version='old_equal_width_6x6',source=str(original.relative_to(ROOT)),source_sha256=sha(original),
        s_edges=grid['x_edges'],a_edges=grid['y_edges'],row_convention='one based; row 1 at bottom (smallest a); column 1 at left (smallest s)',
        expected_targets=24,esoft=1.0,formula='s=0.5*(log(Ex/Esoft)+log(Ey/Esoft)); a=log(Ex/Ey)',
        units='Original FEM modulus units; Esoft=1, Ehard=10; not an independently calibrated physical unit'))
    save(out/'seeds.json',list(range(730000,730010)))
    if not (out/'targets.json').exists():
        save(out/'targets.json',dict(status='blocked_missing_user_selected_cells',grid_sha256=sha(out/'grid.json'),targets=[],
            reason='No authoritative red-circle selection of 24 cells was supplied/found. Diagnostic targets are separate.'))
    p=read(ROOT/'cfm_continuation/run/protocol.json')
    diagnostic=[]
    for i in [0,2,7,14]:
        t=p['development_targets'][i];ex,ey=from_sa(t['s'],t['a'],1.)
        diagnostic.append(dict(target_id=t['target_id'],grid_row=None,grid_col=None,s_target=t['s'],a_target=t['a'],
            Ex_target=float(ex),Ey_target=float(ey),density=t['density'],source='existing continuation development panel'))
    save(out/'diagnostic_targets.json',dict(status='diagnostic_only_not_user_24',targets=diagnostic,source_sha256=sha(ROOT/'cfm_continuation/run/protocol.json')))
    dataset=dict(adapter='adapters.dataset:NumpyMetadataDataset',version='original_8000_v6',directory='cat_binary_dataset',metadata='metadata.csv',array='dataset.npy',
        labels='cfm_v0/run/data.npz',splits='cfm_v0/run/splits.json',training_index_key='train',id_field='filename',image_prefix='images',image_field='filename',
        Ex_field='Ex_app',Ey_field='Ey_app',class_field='requested_style',pixel_hash_field='sha256_pixels',id_numeric_prefix='cat_',
        dedup_field_priority=[['source','source_id'],['source','source_seed'],['source','seed']])
    data=np.load(ROOT/'cfm_v0/run/data.npz')
    bindings={k:sha(ROOT/v) for k,v in dict(metadata='cat_binary_dataset/metadata.csv',array='cat_binary_dataset/dataset.npy',labels='cfm_v0/run/data.npz',splits='cfm_v0/run/splits.json').items()}
    config=dict(schema_version=1,dataset=dataset,targets='evaluation/configs/targets.json',diagnostic_targets='evaluation/configs/diagnostic_targets.json',
        grid='evaluation/configs/grid.json',seeds='evaluation/configs/seeds.json',
        model=dict(adapter='adapters.cfm:CFMAdapter',module_file='cfm_v0/model.py',class_name='ConditionalUNet',architecture=dict(channels=[32,64,128,192,256],emb=256),
            weights_key='ema',dataset_binding=bindings,normalization=dict(mean=data['mean'].tolist(),std=data['std'].tolist()),extra_conditioning={}),
        sampling=dict(method='midpoint',steps=64,shape=[224,224],precision='bfloat16',state_dtype='float32',device='cuda',batch_size=10,
            noise_generator='torch_cuda_generator_per_seed',cpu_threads=4,global_seed=20261117,cudnn_benchmark=False,cudnn_deterministic=True,allow_tf32=True),
        preprocessing=dict(threshold=0.,comparison='>=',raw_representation='x=2*binary-1',black_hard=0,white_soft=1,resize=False,flip=False,
            raw_dtype='float32',binary_dtype='uint8',image_columns='physical x increasing right',image_rows='physical y decreasing down; FEM performs flipud'),
        fem=dict(adapter='adapters.fem:OriginalFEM',module_file='mechanics.py',workers=8,preconditioner_seed=20261117,
            parameters=dict(shape=[224,224],soft_modulus=1.,hard_modulus=10.,poisson=.3,width=40.,height=40.,thickness=2.,strain=.001,solver='amg',tolerance=1e-9,max_iterations=1200)),
        retrieval=dict(top_k=10,distance=dict(name='euclidean_Ex_Ey_fixed_scales',scales=[1.,1.],allow_axis_swap=True,scale_source='fixed original FEM units, no fitting'),
            source_dedup=True,display_orientation='original_unrotated'),
        tolerances=dict(Ex_relative=.05,Ey_relative=.05,s=float(np.log(1.05)),a=float(np.log(1.10)),relative_denominator_floor=1e-12),
        statistics=dict(variance_ddof=1),cache='evaluation/cache')
    for tag,path in [('old_20k','cfm_v0/run/checkpoints/best.pt'),('old_125k','cfm_continuation/run/checkpoints/best.pt')]:
        c=copy.deepcopy(config);c['run_id']=tag;c['model'].update(checkpoint=path,checkpoint_sha256=sha(ROOT/path));save(out/f'{tag}.json',c)
    new=copy.deepcopy(dataset);new.update(version='curated_eight_classes_v1_7857',directory='cat_binary_dataset_curated_v1',labels='cat_binary_dataset_curated_v1/data.npz',
        splits='cat_binary_dataset_curated_v1/splits.json',id_field='new_id',npz_id_key='sample_ids',class_field='group_name_en',id_numeric_prefix=None)
    save(out/'curated_dataset_adapter.json',new)
    print('Frozen configs; formal targets remain pending. Existing targets.json is never overwritten.')

if __name__=='__main__':main()
