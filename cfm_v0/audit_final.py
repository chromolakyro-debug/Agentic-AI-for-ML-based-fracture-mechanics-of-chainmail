"""Acceptance audit of delivered results, independent of the training/metric loops."""
import csv
import hashlib
import json
import re
from collections import Counter
import numpy as np
from PIL import Image
from common import HERE, ROOT, OUT, read_json, save_json, sha_file, S_TOL, A_TOL


def main():
    p=read_json(OUT/'protocol.json');final=OUT/'final'
    manifest=read_json(final/'manifest.json');rows=read_json(final/'samples.json')
    images=np.load(final/'binary.npy',mmap_mode='r');splits=np.load(OUT/'data.npz')
    assert images.shape==(640,224,224) and images.dtype==np.uint8
    assert len(rows)==640 and np.isin(images,[0,1]).all()
    assert Counter(r['target_id'] for r in rows)=={t['target_id']:32 for t in p['final_targets']}
    for target in p['final_targets']:
        group=[r for r in rows if r['target_id']==target['target_id']]
        assert [r['seed'] for r in group]==p['final_noise_seeds']
    for image,row,saved in zip(images,rows,manifest['samples']):
        assert hashlib.sha256(image.tobytes()).hexdigest()==row['sha256_pixels']==saved['sha256_pixels']
        assert row['fem_valid']
        actual=np.array([.5*np.log(row['Ex_app']*row['Ey_app']),np.log(row['Ex_app']/row['Ey_app'])])
        error=actual-[row['s'],row['a']]
        assert np.allclose(error,[row['delta_s'],row['delta_a']],atol=1e-12,rtol=0)
        assert bool(np.all(np.abs(error)<=[S_TOL,A_TOL]))==row['hit_joint']
        f=np.mean(image==0);lo,hi=1/(1-.9*f),1+9*f
        assert lo-1e-6<=row['Ex_app']<=hi+1e-6 and lo-1e-6<=row['Ey_app']<=hi+1e-6
    train=set(splits['train'].tolist())
    for row in read_json(OUT/'retrieval.json')['results']:
        assert row['training_index'] in train and set(row['top32_training_indices'])<=train
    for row in read_json(final/'image_neighbors.json')['results']:
        assert row['pixel_nearest_training_index'] in train
        assert row['shape_nearest_training_index'] in train
        assert set(row['strict_visual_matches_training_indices'])<=train
    assert len(list(csv.DictReader((final/'samples.csv').open(encoding='utf-8-sig',newline=''))))==640
    assert len(list(csv.DictReader((final/'target_summary.csv').open(encoding='utf-8-sig',newline=''))))==20
    assert sha_file(OUT/'protocol.json')==read_json(OUT/'frozen_protocol_sha256.json')['sha256']
    assert sha_file(OUT/'checkpoints'/'best.pt')==manifest['checkpoint_sha256']
    visual=read_json(final/'visual_inspection.json')
    assert visual['manifest_sha256']==sha_file(final/'manifest.json') and len(visual['samples'])==40
    figures=HERE if OUT==(HERE/'run').resolve() else OUT
    for name in ('support_and_targets.png','controllability.png','generated_examples.png'):
        with Image.open(figures/name) as image:assert image.width>=1800 and image.height>=900
    report=(figures/'REPORT.md').read_text(encoding='utf-8')
    assert '**' not in report
    for destination in re.findall(r'\]\(([^)]+)\)',report):
        assert (figures/destination).exists(),destination
    result=dict(status='complete',final_images=640,full_fem_valid=640,joint_hits=sum(r['hit_joint'] for r in rows),
                per_target_count_and_paired_seeds_pass=True,all_binary_hashes_and_fem_rows_match=True,
                retrieval_and_neighbors_training_only=True,protocol_and_checkpoint_unchanged=True,
                visual_review_count=40,three_figures_and_report_links_pass=True)
    save_json(OUT/'final_audit.json',result)
    save_json(OUT/'source_sha256_final.json',{str(path.relative_to(ROOT)):sha_file(path) for path in sorted(HERE.glob('*.py'))})
    review=read_json(final/'visual_review_manifest.json')
    review.update(review_status='completed_actual_pixel_inspection',visual_inspection='visual_inspection.json')
    save_json(final/'visual_review_manifest.json',review)
    completed=read_json(OUT/'evaluation_completed.json')
    if 'wall_seconds_including_wait' in completed:completed['last_completion_invocation_seconds']=completed.pop('wall_seconds_including_wait')
    if 'evaluation_wall_seconds' in completed:completed['sum_successful_command_seconds']=completed.pop('evaluation_wall_seconds')
    completed.update(requires_actual_visual_review=False,visual_review_completed=True,report_ready=True,
                     timing_note='Sum of successful command durations; excludes failed diagnostic attempts and manual report/layout work.')
    save_json(OUT/'evaluation_completed.json',completed)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
