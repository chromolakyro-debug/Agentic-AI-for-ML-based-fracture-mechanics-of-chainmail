"""Independent, exhaustive read-back audit of curated files and original identities."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import time
import numpy as np
from PIL import Image
from curate import ROOT,SOURCE,OUT,OLD,WORK,read,save,sha,row_hash,read_csv


def main():
    start=time.perf_counter()
    source=read(OUT/'provenance'/'source_identity.json')
    rules=read(OUT/'classification_rules.json')['exact_requested_style_mapping']
    assert len(rules)==64
    _,originals=read_csv(SOURCE/'metadata.csv');original={r['filename']:r for r in originals}
    assert len(original)==len(originals)==8000
    identity={r['original_id']:r for r in source['samples']};assert len(identity)==8000
    fields,rows=read_csv(OUT/'metadata.csv');_,indices=read_csv(OUT/'index.csv')
    nested=[json.loads(line) for line in (OUT/'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    nested_by_id={r['curation']['new_id']:r for r in nested}
    assert len(rows)==len(indices)==len(nested)==len(nested_by_id)==7857
    classes=read(OUT/'classes.json')['classes'];class_by_id={r['group_id']:r for r in classes}
    expected={'01':750,'02':1500,'03':750,'04':750,'05':759,'06':766,'07':1500,'08':1082}
    assert Counter(r['group_id'] for r in rows)==expected
    assert {p.name for p in (OUT/'images').iterdir()}=={g+'_'+class_by_id[g]['group_label'] for g in expected}
    assert len({r['new_id'] for r in rows})==len({r['new_relative_path'] for r in rows})==len(rows)
    assert len({r['original_id'] for r in rows})==len(rows)
    active_paths={r['new_relative_path'] for r in rows}
    assert active_paths=={p.relative_to(OUT).as_posix() for p in (OUT/'images').rglob('*') if p.is_file()}
    _,actions=read_csv(OUT/'provenance'/'all_sample_actions.csv')
    action_map={r['original_id']:r for r in actions}
    assert len(actions)==len(action_map)==8000 and set(action_map)==set(identity)
    styles=Counter();style_operations={}
    for oid,action in action_map.items():
        label=action['requested_style'];allowed=rules[label]
        assert action['action']==allowed['action']
        assert action['group_id']==allowed.get('group_id','')
        style_operations.setdefault(label,set()).add((action['action'],action['group_id']))
        styles[label]+=1
    assert all(len(v)==1 for v in style_operations.values()) and len(style_operations)==64
    source_arr=np.load(SOURCE/'dataset.npy',mmap_mode='r',allow_pickle=False)
    arr=np.load(OUT/'dataset.npy',mmap_mode='r',allow_pickle=False)
    d=np.load(OUT/'data.npz',allow_pickle=False);old=np.load(OLD/'data.npz',allow_pickle=False)
    assert arr.shape==(7857,224,224) and arr.dtype==np.uint8
    assert set(d.files)=={'moduli','sa','mean','std','train','validation','test','sample_ids','relative_paths','class_labels','group_ids','group_labels','original_ids','original_cfm_indices','original_array_indices'}
    for k in ['mean','std']:
        assert d[k].dtype==old[k].dtype and d[k].tobytes()==old[k].tobytes()
    local=Counter();all_old_ids=[];observed_hashes=set();ref_hashes={}
    for j,(row,entry) in enumerate(zip(rows,indices)):
        oid=row['original_id'];prior=identity[oid];oldrow=original[prior['original_filename']]
        raw=nested_by_id[row['new_id']]
        assert raw['original_metadata']==oldrow and set(raw['original_metadata'])==set(source['metadata_fields'])
        assert row_hash(oldrow)==prior['original_metadata_sha256']==row['curation.original_metadata_sha256']
        assert row['requested_style']==oldrow['requested_style'] and rules[row['requested_style']]['action']=='keep'
        group=rules[row['requested_style']]['group_id'];c=class_by_id[group];local[group]+=1
        lid=f'cat_{local[group]:06d}';uid=group+'_'+lid
        filename=lid+'.png';relative=c['directory']+'/'+filename
        assert row['group_id']==group and row['new_id']==uid and row['new_filename']==filename and row['new_relative_path']==relative
        assert row['filename']==relative.removeprefix('images/') and (OUT/'images'/row['filename']).resolve()==(OUT/relative).resolve()
        assert row['curation.local_id']==lid and row['group_name_en']==c['group_name_en']
        assert int(row['curation.dataset_index'])==int(entry['dataset_index'])==j
        assert entry['new_id']==uid and entry['original_id']==oid and entry['new_relative_path']==relative
        assert action_map[oid]['new_id']==uid and int(action_map[oid]['new_dataset_index'])==j
        assert all(row[k]==v for k,v in raw['curation'].items())
        assert set(fields)==set(source['metadata_fields'])|set(raw['curation'])
        for k in source['metadata_fields']:
            if k!='filename':assert row[k]==oldrow[k],f'Metadata changed {oid}:{k}'
        assert row['original_filename']==prior['original_filename'] and row['original_relative_path']==prior['original_relative_path']
        path=OUT/relative;source_path=SOURCE/prior['original_relative_path']
        assert sha(path)==sha(source_path)==prior['png_sha256']==entry['png_sha256']
        with Image.open(path) as im:
            assert list(im.size)==prior['size']==[224,224] and im.mode==prior['mode'] and im.format==prior['format']=='PNG'
            pixels=np.asarray(im.convert('L'))
        assert np.isin(pixels,[0,255]).all()
        binary=(pixels//255).astype(np.uint8)
        assert np.array_equal(binary,arr[j]) and np.array_equal(arr[j],source_arr[prior['original_array_index']])
        digest=hashlib.sha256(binary.tobytes()).hexdigest();assert digest==prior['sha256_pixels']==row['sha256_pixels']==entry['sha256_pixels']
        assert digest not in observed_hashes;observed_hashes.add(digest)
        cfmi=prior['original_cfm_index'];assert np.array_equal(d['moduli'][j],old['moduli'][cfmi]) and np.array_equal(d['sa'][j],old['sa'][cfmi])
        assert np.array_equal(d['moduli'][j],[float(oldrow['Ex_app']),float(oldrow['Ey_app'])])
        assert d['sample_ids'][j]==uid and d['relative_paths'][j]==relative and d['original_ids'][j]==oid
        assert d['class_labels'][j]==int(group)-1 and d['group_ids'][j]==group and d['group_labels'][j]==c['group_label']
        assert d['original_cfm_indices'][j]==cfmi and d['original_array_indices'][j]==prior['original_array_index']
        if row['reference_filename']:
            ref=OUT/row['curation.reference_relative_path']
            ref_hashes.setdefault(ref,sha(ref))
            assert ref_hashes[ref]==oldrow['reference_sha256']
        else:assert row['curation.reference_relative_path']==''
        all_old_ids.append((group,prior['original_numeric_id'],prior['original_relative_path']))
        if (j+1)%2000==0:print(f'Read-back image bytes, pixels, complete metadata and FEM association: {j+1}/7857',flush=True)
    assert all_old_ids==sorted(all_old_ids) and dict(local)==expected
    deleted=[json.loads(line) for line in (OUT/'provenance'/'deleted_samples.jsonl').read_text(encoding='utf-8').splitlines()]
    deleted_ids={r['identity']['original_id'] for r in deleted};retained_ids={r['original_id'] for r in rows}
    assert len(deleted)==len(deleted_ids)==143 and not deleted_ids&retained_ids and deleted_ids|retained_ids==set(identity)
    for r in deleted:
        i=r['identity'];assert rules[i['requested_style']]['action']=='delete'
        assert r['original_metadata']==original[i['original_filename']]
        assert sha(SOURCE/i['original_relative_path'])==i['png_sha256']
        assert action_map[i['original_id']]['new_id']==action_map[i['original_id']]['new_relative_path']==''
    splits=read(OUT/'splits.json');seen=set();new_split_of={}
    for key,n in [('train',6285),('validation',786),('test',786)]:
        ids=splits[key];assert len(ids)==n and ids==sorted(ids) and len(set(ids))==n and not seen&set(ids)
        seen.update(ids);assert np.array_equal(d[key],ids)
        assert splits['sample_ids'][key]==[rows[i]['new_id'] for i in ids]
        assert splits['sample_paths'][key]==[rows[i]['new_relative_path'] for i in ids]
        assert all(rows[i]['curation.split']==indices[i]['split']==key for i in ids)
        new_split_of.update({i:key for i in ids})
        for g,total in expected.items():
            count=sum(rows[i]['group_id']==g for i in ids)
            assert count==splits['class_counts'][g][key]
            assert abs(count-total*(.8 if key=='train' else .1))<2
    assert seen==set(range(7857))
    assert not set(d['original_ids'])&deleted_ids
    filtered=read(OUT/'provenance'/'filtered_original_splits.json');migration=read(OUT/'provenance'/'split_migration.json')
    oldsplit=read(OLD/'splits.json');matrix={a:{b:0 for b in ('train','validation','test')} for a in ('train','validation','test')}
    for key in matrix:
        expected_ids=[i for i,r in enumerate(rows) if identity[r['original_id']]['original_cfm_index'] in set(oldsplit[key])]
        assert filtered[key]==expected_ids
        for i in expected_ids:
            assert rows[i]['curation.original_split']==indices[i]['original_split']==key
            matrix[key][new_split_of[i]]+=1
    assert matrix==migration['matrix'] and sum(matrix[a][b] for a in matrix for b in matrix if a!=b)==migration['migrated']
    legacy=np.load(OUT/'provenance'/'retained_legacy_sampling_and_density.npz',allow_pickle=False)
    for key in ['train','validation']:
        old_positions={int(i):j for j,i in enumerate(old[key])}
        for oid,new_index in zip(legacy[key+'_original_ids'],legacy[key+'_new_indices']):assert rows[new_index]['original_id']==oid
        pos=[old_positions[identity[oid]['original_cfm_index']] for oid in legacy[key+'_original_ids']]
        assert np.array_equal(legacy[key+'_r20'],old[key+'_r20'][pos])
        if key=='train':assert np.array_equal(legacy['train_weights'],old['train_weights'][pos])
    for key in ['mean','std','cell_counts','bin_edges_s','bin_edges_a']:assert np.array_equal(legacy[key],old[key])
    assert sha(OUT/'provenance'/'original_dataset'/'metadata.csv')==sha(SOURCE/'metadata.csv')
    for name in ['data.npz','splits.json','protocol.json','data_audit.json']:assert sha(OUT/'provenance'/'cfm_v0'/name)==sha(OLD/name)
    protected_count=0;changed_code=[]
    for file in source['source_files']:
        after=sha(ROOT/file['path'])
        if after!=file['sha256']:
            assert file.get('allowed_change') and file['path']=='cfm_v0/train.py'
            assert sha(OUT/'provenance'/'source_code_before'/'train.py')==file['sha256']
            changed_code.append(dict(path=file['path'],before=file['sha256'],after=after,reason=file['allowed_change']))
        else:protected_count+=1
    assert not list(OUT.rglob('*.tmp')) and not list(OUT.rglob('*.pt'))
    result=dict(status='passed',source_count=8000,retained=7857,deleted=143,source_fine_classes=64,
        keep_fine_classes=sum(v['action']=='keep' for v in rules.values()),delete_fine_classes=sum(v['action']=='delete' for v in rules.values()),
        class_counts=expected,all_original_samples_exactly_one_action=True,no_per_image_classification=True,
        all_retained_file_sha256_equal=True,all_pixels_shape_phase_equal=True,
        all_124_original_metadata_fields_exactly_preserved=True,all_Ex_Ey_sa_values_exactly_preserved=True,
        unchanged_normalization=True,all_active_references_valid=True,no_deleted_sample_in_active_index_or_split=True,
        new_ids_unique_and_class_local_numbers_contiguous=True,
        splits=dict(counts={k:len(splits[k]) for k in matrix},disjoint=True,class_stratified=True,migrated_from_old=migration['migrated'],
            filter_only_criterion='Superseded by explicit user clarification; old membership archived separately'),
        original_images_preserved=8000,original_other_files_hash_unchanged=protected_count,loader_code_changes=changed_code,
        no_FEM_recalculation=True,no_training=True,elapsed_seconds=time.perf_counter()-start)
    save(OUT/'validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
