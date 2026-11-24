"""Audit and curate all original samples without image encoding, FEM, or training."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from rules import CLASSES, KEEP, DELETE, SEED, rules_document

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'cat_binary_dataset'
OUT=ROOT/'cat_binary_dataset_curated_v1'
WORK=HERE/'work'
OLD=ROOT/'cfm_v0'/'run'
EXTRA=('new_id','new_filename','new_relative_path','group_id','group_name_en',
       'original_id','original_filename','original_relative_path',
       'curation.dataset_index','curation.local_id','curation.class_label',
       'curation.group_label','curation.group_name_zh','curation.original_numeric_id',
       'curation.original_array_index','curation.original_cfm_index',
       'curation.original_split','curation.split','curation.original_metadata_sha256',
       'curation.reference_relative_path')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def row_hash(row):return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
    assert len(fields)==len(set(fields)) and all(set(r)==set(fields) for r in rows)
    return fields,rows
def csv_out(path,fields,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def jsonl_out(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def audit():
    WORK.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter();fields,rows=read_csv(SOURCE/'metadata.csv')
    rules=rules_document();save(HERE/'classification_rules.json',rules)
    labels=Counter(r['requested_style'] for r in rows)
    unmatched=sorted(set(labels)-set(KEEP)-DELETE)
    missing=sorted((set(KEEP)|DELETE)-set(labels))
    counts=Counter(KEEP[r['requested_style']] for r in rows if r['requested_style'] in KEEP)
    observed=dict(source_count=len(rows),source_fine_classes=len(labels),retained_classes=len(set(labels)&set(KEEP)),
        deleted_classes=len(set(labels)&DELETE),counts=dict(sorted(counts.items())),
        retained=sum(counts.values()),deleted=sum(labels[k] for k in DELETE),unmatched_labels=unmatched,
        mapped_but_absent_labels=missing,label_counts=dict(sorted(labels.items())))
    save(WORK/'mapping_preflight.json',observed)
    assert not unmatched and not missing, 'Unexpected labels; inspect mapping_preflight.json before proceeding'
    assert len(rows)==8000 and len(labels)==64 and observed['deleted']==143
    assert all(counts[g]==n for g,zh,en,tag,n in CLASSES) and sum(counts.values())==7857
    assert not set(fields)&set(EXTRA), 'Added field collision: implement a distinct curation namespace before writing'
    arr=np.load(SOURCE/'dataset.npy',mmap_mode='r',allow_pickle=False)
    assert arr.shape==(8000,224,224) and arr.dtype==np.uint8
    # Establish array identity by pixel SHA, independently of CSV order and directory order.
    hash_to_array={}
    for i,image in enumerate(arr):
        assert np.isin(image,[0,1]).all()
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        assert digest not in hash_to_array, 'Ambiguous duplicate pixels in source array'
        hash_to_array[digest]=i
    protocol=read(OLD/'protocol.json');old_audit=read(OLD/'data_audit.json');splits=read(OLD/'splits.json')
    assert protocol['data_sha256']==sha(SOURCE/'dataset.npy') and protocol['metadata_sha256']==sha(SOURCE/'metadata.csv')
    assert old_audit['all_png_array_metadata_hashes_match'] and old_audit['n']==8000
    frozen=np.load(OLD/'data.npz',allow_pickle=False)
    split_of={}
    for key,n in [('train',6400),('validation',800),('test',800)]:
        assert len(splits[key])==n and np.array_equal(frozen[key],splits[key])
        for i in splits[key]:assert i not in split_of;split_of[i]=key
    assert set(split_of)==set(range(8000))
    for name in ('data.npz','splits.json'):
        assert sha(OLD/name)==sha(ROOT/'cfm_continuation'/'run'/name)
    identities=[];used_ids=set();used_paths=set();used_array=set()
    for csv_index,row in enumerate(rows):
        match=re.fullmatch(r'cat_(\d+)\.png',row['filename'])
        assert match, f'Unrecognized original sample ID: {row["filename"]}'
        numeric=int(match.group(1));relative='images/'+row['filename']
        assert numeric not in used_ids and relative not in used_paths
        used_ids.add(numeric);used_paths.add(relative)
        path=SOURCE/relative
        with Image.open(path) as im:
            size=im.size;mode=im.mode;fmt=im.format;pixels=np.array(im.convert('L'))
        assert size==(224,224) and fmt=='PNG' and np.isin(pixels,[0,255]).all()
        binary=(pixels//255).astype(np.uint8);digest=hashlib.sha256(binary.tobytes()).hexdigest()
        assert digest==row['sha256_pixels'] and digest in hash_to_array
        array_index=hash_to_array[digest];assert array_index not in used_array;used_array.add(array_index)
        assert np.array_equal(binary,arr[array_index])
        assert abs(float(row['black_fraction'])-(binary==0).mean())<1e-12
        # The frozen original CFM protocol authenticates both source files. Its
        # documented ID-index rule is now additionally checked for every sample.
        assert numeric==csv_index==array_index and row['filename']==f'cat_{numeric:04d}.png'
        values=np.array([float(row['Ex_app']),float(row['Ey_app'])],dtype=np.float64)
        assert np.array_equal(values,frozen['moduli'][numeric])
        expected_sa=np.array([.5*(np.log(values[0]/float(row['E_soft']))+np.log(values[1]/float(row['E_soft']))),np.log(values[0]/values[1])])
        assert np.array_equal(expected_sa,frozen['sa'][numeric])
        assert float(row['E_soft'])==1 and float(row['E_hard'])==10 and row['fem_valid'].lower()=='true'
        identities.append(dict(original_id=Path(row['filename']).stem,original_numeric_id=numeric,
            original_filename=row['filename'],original_relative_path=relative,original_csv_index=csv_index,
            original_array_index=array_index,original_cfm_index=numeric,original_split=split_of[numeric],
            requested_style=row['requested_style'],action='keep' if row['requested_style'] in KEEP else 'delete',
            group_id=KEEP.get(row['requested_style']),png_sha256=sha(path),sha256_pixels=digest,
            mode=mode,size=list(size),format=fmt,original_metadata_sha256=row_hash(row)))
        if (csv_index+1)%2000==0:print(f'Identity/PNG/NPY/metadata/FEM association verified: {csv_index+1}/8000',flush=True)
    assert {p.relative_to(SOURCE).as_posix() for p in (SOURCE/'images').iterdir() if p.is_file()}==used_paths
    # Protect the source release and all existing checkpoints/logs. Results are not migrated.
    protected=[]
    for base in [SOURCE,OLD,ROOT/'cfm_continuation'/'run']:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
            if base!=SOURCE and ('code_snapshots' in p.parts):continue
            if base==SOURCE and p.parent==SOURCE/'images':continue # full per-image identities above
            protected.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    for p in [ROOT/'cfm_v0'/'train.py',ROOT/'cfm_v0'/'common.py',ROOT/'mechanics.py',ROOT/'evaluate_mechanics.py']:
        protected.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),
            allowed_change='optional dataset/run path arguments to existing loader only' if p.name=='train.py' else None))
    snapshots=WORK/'source_code_before';snapshots.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'cfm_v0'/'train.py',snapshots/'train.py')
    shutil.copy2(ROOT/'cfm_v0'/'common.py',snapshots/'common.py')
    save(WORK/'source_identity.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        metadata_fields=fields,source_files=protected,samples=identities,
        row_association='Explicit filename numeric ID + unique pixel SHA join to NPY + authenticated CFM protocol and exact per-ID Ex/Ey/s/a comparisons',
        observed=observed,elapsed_seconds=time.perf_counter()-tick))
    print(json.dumps(observed|dict(elapsed_seconds=time.perf_counter()-tick),ensure_ascii=False),flush=True)


def build():
    tick=time.perf_counter();audit_doc=read(WORK/'source_identity.json');rules=read(HERE/'classification_rules.json')
    assert rules==rules_document()
    if (OUT/'STATUS.json').exists():assert read(OUT/'STATUS.json')['status']!='complete','Preserve completed dataset'
    OUT.mkdir(exist_ok=True);save(OUT/'STATUS.json',dict(status='building',not_for_training=True))
    fields,rows=read_csv(SOURCE/'metadata.csv')
    for name in ('dataset.npy','metadata.csv'):
        expected=next(r['sha256'] for r in audit_doc['source_files'] if r['path']=='cat_binary_dataset/'+name)
        assert sha(SOURCE/name)==expected
    records=audit_doc['samples'];assert all(row_hash(rows[r['original_csv_index']])==r['original_metadata_sha256'] for r in records)
    retained=sorted([r for r in records if r['action']=='keep'],key=lambda r:(r['group_id'],r['original_numeric_id'],r['original_relative_path']))
    deleted=[r for r in records if r['action']=='delete'];n=len(retained)
    labels=np.array([int(r['group_id'])-1 for r in retained],dtype=np.int64)
    tr,rest=train_test_split(np.arange(n,dtype=np.int64),test_size=.2,stratify=labels,random_state=SEED)
    va,te=train_test_split(rest,test_size=.5,stratify=labels[rest],random_state=SEED+1)
    partitions={k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}
    split_of={int(i):key for key,ids in partitions.items() for i in ids}
    classes={g:dict(group_id=g,group_name_zh=zh,group_name_en=en,group_label=tag,
                   directory='images/'+g+'_'+tag,class_label=int(g)-1,count=count) for g,zh,en,tag,count in CLASSES}
    for c in classes.values():(OUT/c['directory']).mkdir(parents=True,exist_ok=True)
    provenance=OUT/'provenance';provenance.mkdir(exist_ok=True)
    for path in ['metadata.csv','README.md','quality_report.md']:
        dst=provenance/'original_dataset'/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/path,dst)
    for name in ['data.npz','splits.json','protocol.json','data_audit.json']:
        dst=provenance/'cfm_v0'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,dst)
    for src in sorted((WORK/'source_code_before').glob('*.py')):
        dst=provenance/'source_code_before'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
    references=provenance/'style_references';references.mkdir(exist_ok=True)
    for src in sorted((ROOT/'cat-style-references').glob('*.png')):shutil.copy2(src,references/src.name)
    shutil.copy2(HERE/'classification_rules.json',OUT/'classification_rules.json')
    shutil.copy2(WORK/'source_identity.json',provenance/'source_identity.json')
    source_arrays=np.load(SOURCE/'dataset.npy',mmap_mode='r',allow_pickle=False)
    output_arrays=np.lib.format.open_memmap(OUT/'dataset.npy',mode='w+',dtype=np.uint8,shape=(n,224,224))
    metadata=[];nested=[];index_rows=[];decisions=[];class_counter=Counter();copy_hashes=[]
    for index,record in enumerate(retained):
        r=rows[record['original_csv_index']];gid=record['group_id'];c=classes[gid]
        class_counter[gid]+=1;local=f'cat_{class_counter[gid]:06d}';new_id=gid+'_'+local
        filename=local+Path(record['original_filename']).suffix;relative=c['directory']+'/'+filename
        src=SOURCE/record['original_relative_path'];dst=OUT/relative
        shutil.copy2(src,dst);digest=sha(dst);assert digest==record['png_sha256']
        output_arrays[index]=source_arrays[record['original_array_index']]
        new=dict(new_id=new_id,new_filename=filename,new_relative_path=relative,group_id=gid,
            group_name_en=c['group_name_en'],original_id=record['original_id'],original_filename=record['original_filename'],
            original_relative_path=record['original_relative_path'])
        extra={'curation.dataset_index':str(index),'curation.local_id':local,'curation.class_label':str(c['class_label']),
            'curation.group_label':c['group_label'],'curation.group_name_zh':c['group_name_zh'],
            'curation.original_numeric_id':str(record['original_numeric_id']),
            'curation.original_array_index':str(record['original_array_index']),
            'curation.original_cfm_index':str(record['original_cfm_index']),
            'curation.original_split':record['original_split'],'curation.split':split_of[index],
            'curation.original_metadata_sha256':record['original_metadata_sha256'],
            'curation.reference_relative_path':'provenance/style_references/'+r['reference_filename'] if r['reference_filename'] else ''}
        active=dict(r);active['filename']=relative.removeprefix('images/');active.update(new);active.update(extra)
        metadata.append(active)
        nested.append(dict(curation=dict(new,**extra),original_metadata=r))
        index_rows.append(dict(dataset_index=index,**new,class_label=c['class_label'],split=split_of[index],
            original_split=record['original_split'],original_array_index=record['original_array_index'],
            sha256_pixels=record['sha256_pixels'],png_sha256=digest))
        decisions.append(dict(original_id=record['original_id'],original_numeric_id=record['original_numeric_id'],
            original_relative_path=record['original_relative_path'],requested_style=r['requested_style'],action='keep',
            group_id=gid,new_id=new_id,new_relative_path=relative,new_dataset_index=index,
            original_split=record['original_split'],new_split=split_of[index],png_sha256=digest,
            original_metadata_sha256=record['original_metadata_sha256']))
        copy_hashes.append(dict(path=relative,bytes=dst.stat().st_size,sha256=digest))
        if (index+1)%2000==0:print(f'Copied byte-identical retained files: {index+1}/{n}',flush=True)
    output_arrays.flush();del output_arrays
    csv_out(OUT/'metadata.csv',fields+list(EXTRA),metadata)
    jsonl_out(OUT/'metadata.jsonl',nested)
    csv_out(OUT/'index.csv',list(index_rows[0]),index_rows)
    for r in deleted:
        decisions.append(dict(original_id=r['original_id'],original_numeric_id=r['original_numeric_id'],
            original_relative_path=r['original_relative_path'],requested_style=r['requested_style'],action='delete',
            group_id='',new_id='',new_relative_path='',new_dataset_index='',original_split=r['original_split'],new_split='',
            png_sha256=r['png_sha256'],original_metadata_sha256=r['original_metadata_sha256']))
    decisions.sort(key=lambda r:r['original_numeric_id'])
    csv_out(provenance/'all_sample_actions.csv',list(decisions[0]),decisions)
    jsonl_out(provenance/'deleted_samples.jsonl',[
        dict(action='delete',reason='User-requested whole requested_style exclusion',identity=r,
             original_metadata=rows[r['original_csv_index']]) for r in deleted])
    deleted_flat=[]
    for r in deleted:
        item=dict(rows[r['original_csv_index']]);item.update({'curation.original_id':r['original_id'],
            'curation.original_relative_path':r['original_relative_path'],'curation.original_split':r['original_split'],
            'curation.action':'delete','curation.png_sha256':r['png_sha256']});deleted_flat.append(item)
    csv_out(provenance/'deleted_samples.csv',list(deleted_flat[0]),deleted_flat)
    group_split_counts={g:{k:int(np.sum(labels[ids]==int(g)-1)) for k,ids in partitions.items()} for g in classes}
    splitdoc=dict(seed=SEED,method='Two sklearn stratified splits by final class: test_size=.2 then .5; random_state=seed,seed+1',
        ratio=[.8,.1,.1],rounding='Integer allocation by stratified split; 6285/786/786 total, approximate80/10/10 in each class',
        **{k:v.tolist() for k,v in partitions.items()},
        sample_ids={k:[index_rows[i]['new_id'] for i in ids] for k,ids in partitions.items()},
        sample_paths={k:[index_rows[i]['new_relative_path'] for i in ids] for k,ids in partitions.items()},
        class_counts=group_split_counts,user_instruction_resolution=rules['split']['user_resolution'])
    save(OUT/'splits.json',splitdoc)
    filtered={k:[i for i,r in enumerate(retained) if r['original_split']==k] for k in partitions}
    save(provenance/'filtered_original_splits.json',dict(active=False,note='Original membership, only filtering/remapping; historical comparison, NOT active resplit',
        **filtered,sample_ids={k:[index_rows[i]['new_id'] for i in ids] for k,ids in filtered.items()}))
    matrix={old:{new:sum(r['original_split']==old and split_of[i]==new for i,r in enumerate(retained)) for new in partitions} for old in partitions}
    save(provenance/'split_migration.json',dict(matrix=matrix,migrated=sum(r['original_split']!=split_of[i] for i,r in enumerate(retained)),
        reason='User explicitly selected new class-stratified80/10/10; no overlap between active sets',
        filtered_old_counts={k:len(v) for k,v in filtered.items()},new_counts={k:len(v) for k,v in partitions.items()}))
    frozen=np.load(OLD/'data.npz',allow_pickle=False);cfm_ids=np.array([r['original_cfm_index'] for r in retained],dtype=np.int64)
    np.savez(OUT/'data.npz',moduli=frozen['moduli'][cfm_ids],sa=frozen['sa'][cfm_ids],mean=frozen['mean'],std=frozen['std'],
        **partitions,sample_ids=np.array([r['new_id'] for r in index_rows]),
        relative_paths=np.array([r['new_relative_path'] for r in index_rows]),class_labels=labels,
        group_ids=np.array([r['group_id'] for r in retained]),group_labels=np.array([classes[r['group_id']]['group_label'] for r in retained]),
        original_ids=np.array([r['original_id'] for r in retained]),original_cfm_indices=cfm_ids,
        original_array_indices=np.array([r['original_array_index'] for r in retained],dtype=np.int64))
    # Old training-only weights/densities have no values for old val/test. Preserve
    # them joined by original ID, rather than fabricate active new-training weights.
    retained_lookup={r['original_cfm_index']:i for i,r in enumerate(retained)}
    legacy={}
    for split in ['train','validation']:
        positions=[j for j,i in enumerate(frozen[split]) if int(i) in retained_lookup]
        old_ids=frozen[split][positions]
        legacy[split+'_original_ids']=np.array([f'cat_{int(i):04d}' for i in old_ids])
        legacy[split+'_new_indices']=np.array([retained_lookup[int(i)] for i in old_ids],dtype=np.int64)
        legacy[split+'_r20']=frozen[split+'_r20'][positions]
        if split=='train':legacy['train_weights']=frozen['train_weights'][positions]
    for key in ['mean','std','cell_counts','bin_edges_s','bin_edges_a']:legacy[key]=frozen[key]
    np.savez(provenance/'retained_legacy_sampling_and_density.npz',**legacy)
    save(OUT/'normalization.json',dict(mean=frozen['mean'].tolist(),std=frozen['std'].tolist(),
        preserved_exact=True,source='provenance/cfm_v0/data.npz',fit_population='Original6400 CFM training samples, NOT the newly stratified training set',
        definitions=dict(s='0.5*(log(Ex/Esoft)+log(Ey/Esoft))',a='log(Ex/Ey)',Esoft=1.,Ehard=10.,black_value=0,white_value=1),
        note='User requested unchanged normalization. Some new validation/test samples were in original training; these frozen statistics and old checkpoints do not constitute a fresh unbiased holdout. New training sampler weights/support are not computed in this curation task.'))
    save(OUT/'classes.json',dict(classes=list(classes.values()),counts={g:class_counter[g] for g in classes},
        label_encoding='class_labels=group_id-1 (0..7); group_id=01..08; global new_id=group_id + underscore + class-local ID'))
    save(OUT/'schema.json',dict(version=1,n=n,metadata_columns=fields+list(EXTRA),original_metadata_columns=fields,
        active_files=['dataset.npy','metadata.csv','metadata.jsonl','index.csv','data.npz','splits.json','classes.json'],
        filename_semantics='metadata.csv filename is relative to images/; new_relative_path is relative to dataset root',
        original_metadata='metadata.jsonl original_metadata contains every original field/value unchanged, including the original filename; provenance/original_dataset/metadata.csv is also byte-identical',
        additional_fields=list(EXTRA),split='Active splits are newly class-stratified, old splits are provenance only',
        numeric_identity='Seeds and all original CSV values remain strings; NPY/NPZ numeric values copied without recomputation',
        not_a_training_run='No optimizer/checkpoint/training is created. Active data.npz omits stale old train_weights/support distances; full originals and filtered history are in provenance.'))
    save(provenance/'image_file_hashes.json',copy_hashes)
    save(provenance/'curation_summary.json',dict(kept=n,deleted=len(deleted),classes=8,source_fine_classes=64,
        retained_fine_classes=51,deleted_fine_classes=13,split_counts={k:len(v) for k,v in partitions.items()},
        group_split_counts=group_split_counts,elapsed_seconds=time.perf_counter()-tick,
        source_audit_elapsed_seconds=audit_doc['elapsed_seconds'],no_fem_recalculation=True,no_training=True,no_image_reencoding=True))
    save(OUT/'STATUS.json',dict(status='awaiting_validation',not_for_training=True))
    print(json.dumps(read(provenance/'curation_summary.json'),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','build']);args=p.parse_args()
    audit() if args.phase=='audit' else build()
