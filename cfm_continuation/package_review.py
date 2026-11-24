"""Compact whitelist-based audit ZIP; no full training images or model weights."""
import paths
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime,timezone
from pathlib import Path
from common import OUT,read_json,save_json,sha_file
from package_ai_review import VERIFY_SCRIPT


def main():
    stage=paths.HERE/'ai_review_package';destination=paths.ROOT/'cfm_continuation_ai_review.zip';prefix='cfm_continuation_ai_review/'
    assert not stage.exists() and not destination.exists(), 'Preserve existing archives; use a new destination'
    assert (paths.HERE/'REPORT.md').exists() and read_json(OUT/'numerical_audit.json')['status']=='passed'
    assert read_json(OUT/'checkpoint_inventory.json')['original_files_hash_verified_unchanged']>0
    stage.mkdir();provenance={}
    def copy(f,relative=None):
        f=Path(f);relative=relative or f.relative_to(paths.ROOT).as_posix();dest=stage/relative
        assert f.suffix not in ('.pt','.ckpt','.pkl'),f
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        provenance[relative]=dict(source=str(f.resolve()),sha256=sha_file(f),operation='byte-for-byte copy')
    copy(paths.HERE/'START_HERE.md','START_HERE.md')
    for f in sorted(paths.HERE.glob('*.py')):copy(f)
    for name in ('REPORT.md','CODE_CHANGES.md','REPRODUCE.md','evaluation_rules.json','training_curves.png','support_and_targets.png','controllability_comparison.png','generated_examples.png','training_neighbor_examples.png'):
        copy(paths.HERE/name)
    for f in sorted((paths.HERE/'reviews').glob('*.json')):copy(f)
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.suffix in ('.json','.jsonl','.csv','.txt') and f.name!='complete_training_history.json':copy(f)
    copy(OUT/'data.npz')
    for f in sorted((OUT/'code_snapshots').rglob('*')):
        if f.is_file():copy(f)
    for folder in sorted(OUT.glob('development_*'))+[OUT/'final']:
        if not folder.is_dir():continue
        for f in sorted(folder.iterdir()):
            if f.is_file() and (f.suffix in ('.json','.jsonl','.csv','.png') or f.name=='binary.npy'):copy(f)
        for f in sorted((folder/'visual_details').glob('*.png')):copy(f)
    for name in ('common.py','model.py','train.py','generate.py','fem.py','metrics.py','diagnostics.py','plots.py','prepare.py','package_ai_review.py','requirements.txt'):
        copy(paths.ROOT/'cfm_v0'/name)
    for name in ('mechanics.py','evaluate_mechanics.py','mechanics_selection.py','reference_style_selection.py','augmentation_selection.py'):
        copy(paths.ROOT/name)
    for name in ('training.jsonl','train_config.json','training_summary.json','environment.json','protocol.json','coverage_diagnostics.json','retrieval.json'):
        copy(paths.ORIGINAL/name)
    for name in ('binary.npy','manifest.json','metrics.json','samples.csv','samples.json','target_summary.csv','fem.jsonl','fem_summary.json','image_neighbors.json','visual_inspection.json','visual_review_fixed_seeds.png'):
        copy(paths.ORIGINAL/'final'/name)
    copy(paths.ROOT/'cfm_v0'/'ai_review_package'/'context'/'dataset_properties_8000.csv','context/dataset_properties_8000.csv')
    copy(paths.ROOT/'cat_binary_dataset'/'README.md','context/DATASET_README.md')
    copy(paths.ROOT/'cat_binary_dataset'/'quality_report.md','context/DATASET_QUALITY_REPORT.md')
    # Small, original-pixel train exemplars for selected generated/nearest comparisons.
    chosen=read_json(OUT/'final'/'figure_examples.json')['keys'];nn={r['key']:r for r in read_json(OUT/'final'/'image_neighbors.json')['results']}
    retrieval={r['target_id']:r for r in read_json(OUT/'retrieval.json')['results']};indices=set();comparisons=[]
    properties=list(csv.DictReader((paths.ROOT/'cfm_v0'/'ai_review_package'/'context'/'dataset_properties_8000.csv').open(encoding='utf-8-sig')))
    for key in chosen:
        r=nn[key];attribute=retrieval[r['target_id']]['training_index'];shape=r.get('shape_nearest_training_index');pixel=r['pixel_nearest_training_index']
        ids=[attribute,pixel]+([shape] if shape is not None else []);indices.update(ids)
        comparisons.append(dict(key=key,attribute_nearest_training_index=attribute,pixel_nearest_training_index=pixel,shape_nearest_training_index=shape,
                                training_properties=[properties[i] for i in sorted(set(ids))]))
    for r in read_json(OUT/'final'/'neighbor_examples.json')['records']:
        indices.update([r['attribute_training_index'],r['shape_training_index']])
    for i in sorted(indices):copy(paths.ROOT/'cat_binary_dataset'/'images'/f'cat_{i:04d}.png',f'context/selected_training_images/cat_{i:04d}.png')
    save_json(stage/'context'/'selected_training_neighbors.json',dict(selection='Same explicitly marked representative keys as generated_examples.png; these examples are not a random sample or proof of novelty.',results=comparisons))
    save_json(stage/'SOURCE_PROVENANCE.json',provenance)
    verifier=VERIFY_SCRIPT.replace("prefix='cfm_v0_ai_review/'","prefix='cfm_continuation_ai_review/'")
    (stage/'VERIFY_ARCHIVE.py').write_text(verifier,encoding='utf-8')
    files=[dict(path=f.relative_to(stage).as_posix(),bytes=f.stat().st_size,sha256=sha_file(f)) for f in sorted(stage.rglob('*')) if f.is_file()]
    save_json(stage/'AUDIT_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files,
        excluded=['Full 8000 training maps','All .pt model weights and optimizer tensors','Descriptor caches','Environment caches','Repeated per-target generation chunks'],
        checkpoint_inventory='cfm_continuation/run/checkpoint_inventory.json'))
    hashes=''.join(f"{r['sha256']}  {r['path']}\n" for r in files)+f"{sha_file(stage/'AUDIT_MANIFEST.json')}  AUDIT_MANIFEST.json\n"
    (stage/'SHA256SUMS.txt').write_text(hashes,encoding='utf-8')
    temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in sorted(stage.rglob('*')):
            if f.is_file():z.write(f,prefix+f.relative_to(stage).as_posix())
    result=subprocess.run([sys.executable,str(stage/'VERIFY_ARCHIVE.py'),str(temporary)],capture_output=True,text=True,check=True)
    validation=json.loads(result.stdout)
    # Verify the staged package itself has every required input for independent number checks.
    numerical=subprocess.run([sys.executable,str(stage/'cfm_continuation'/'verify_results.py'),'--run',str(stage/'cfm_continuation'/'run')],capture_output=True,text=True,check=True)
    validation['staged_numerical_audit']=json.loads(numerical.stdout)
    assert temporary.stat().st_size<50*1024**2,'Unexpectedly large review package; inspect whitelist'
    temporary.replace(destination);hashvalue=sha_file(destination)
    destination.with_suffix('.zip.sha256').write_text(hashvalue+'  '+destination.name+'\n',encoding='utf-8')
    save_json(paths.HERE/'archive_validation.json',validation|dict(path=str(destination),bytes=destination.stat().st_size,sha256=hashvalue))
    print(json.dumps({'path':str(destination),'bytes':destination.stat().st_size,'sha256':hashvalue,'files':len(files)},indent=2))

if __name__=='__main__':main()
