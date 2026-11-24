"""Finish authorized staging work after generation; never publish the live dataset.

Consumes only explicit completed AI review findings, waits for existing FEM and
GPU jobs, then performs deterministic selection, export, checks and analysis.
Any shortfall/check failure stops before publication. Final plot/preview review
and guarded publication remain the root agent's responsibility.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import record_reference_findings as findings
import reference_style_workflow as workflow
import reference_style_candidates as engine
import assemble_reference_generator as bundle
from refine_release import choose_style_indices,run as refine

ROOT=Path(__file__).resolve().parent


def load(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def command(*values):
    print('RUN: '+' '.join(map(str,values)),flush=True)
    subprocess.run([sys.executable,*map(str,values)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--total',type=int,default=8000)
    args=p.parse_args();w=args.work_dir
    phase=w/'finishing_state.json'
    def state(name,**data):
        workflow.write_json(phase,dict(stage=name,**data));print('STAGE '+name,flush=True)
    state('waiting_for_generation_fem_and_ai_reviews')
    last=None
    while True:
        newly=findings.record_all(w)
        gen=load(w/'full_generation_state.json',{})
        index=load(w/'source_review_index.json',{})
        reviews=load(w/'reference_visual_review.json',{})
        label_path=w/'full_labels.jsonl'
        label_count=sum(1 for _ in label_path.open(encoding='utf-8')) if label_path.exists() else 0
        reviewed=len(reviews.get('source_review_pages',{}))
        status=(index.get('next_source_index',0)//100,label_count//100,reviewed)
        if status!=last:
            print(json.dumps(dict(processed_sources=index.get('next_source_index',0),
                target_sources=gen.get('source_count'),FEM_labels=label_count,
                sealed_pages=index.get('sealed_page_count',0),AI_reviewed_pages=reviewed,
                passed_pixel_hashes=len(reviews.get('passed_hashes',[])),
                excluded_pixel_hashes=len(reviews.get('excluded_hashes',{})))),flush=True)
            last=status
        if gen.get('complete') and index.get('generation_complete'):
            if reviewed==index.get('sealed_page_count') and label_count>=gen['valid_sources']:
                break
        time.sleep(30)
    state('selecting')
    command('reference_style_workflow.py','select-full','--work-dir',w,'--total',args.total)
    selected=load(w/'full_selection.json')
    if selected['selected_count']!=args.total-1300 or selected['visual_review']!='pass':
        raise RuntimeError('Selection shortfall or review incomplete; no padding or automatic quota transfer')
    state('exporting_staging')
    command('reference_style_workflow.py','export','--work-dir',w,'--total',args.total)
    state('checking_mesh_refinement')
    release=w/'release'
    images=np.load(release/'dataset.npy',mmap_mode='r',allow_pickle=False)
    with (release/'metadata.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    ids,reasons,descriptors=choose_style_indices(images,rows,1250,families=engine.STYLE_FAMILIES)
    workflow.write_json(w/'refinement_plan.json',dict(indices=ids,reasons=reasons,descriptors=descriptors))
    refine(release,w/'refinement.json',fail_on_large_change=True,first_index=1250,indices=ids)
    del images
    state('checking_same_workstation_reproduction')
    command('check_reference_reproduction.py','--work-dir',w,'--dry-run')
    command('check_reference_reproduction.py','--work-dir',w)
    state('packaging_and_analysis')
    # Refresh the concise report with completed actual numeric evidence.
    cfg=SimpleNamespace(work_dir=w,output=release)
    workflow.write_report(cfg,rows,load(w/'final_pair_audit.json'))
    command('assemble_reference_generator.py',release/'generate_dataset.py','--work-dir',w)
    analysis=ROOT/'analysis'/'reference_expansion'
    command('analysis/compare_reference_expansion.py','--base',w/'base1250','--dataset',release,'--output-dir',analysis)
    command('analysis/show_reference_styles.py','--dataset',release,'--output-dir',analysis)
    command(release/'generate_dataset.py','validate','--output',release,'--total',args.total)
    state('staging_ready_for_final_visual_review',count=args.total,release=str(release),analysis=str(analysis),
          publication_performed=False)


if __name__=='__main__':main()
