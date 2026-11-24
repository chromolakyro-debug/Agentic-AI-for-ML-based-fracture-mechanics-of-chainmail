"""Audit fresh standalone export with embedded review, outside the source project.

Reuses immutable cached candidates/FEM labels, not newly generated GPU samples.
All copies/links and the second export remain in scratch. Does not publish data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def link_readonly_reference(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(source)!=sha(destination):raise ValueError(f'Audit destination differs: {destination}')
    else:
        os.link(source,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--total',type=int,default=8000)
    args=p.parse_args();w=args.work_dir.resolve();source=w/'release'
    audit=w/'standalone_export_audit';work=audit/'work';original=audit/'base1250';output=audit/'export'
    audit.mkdir(exist_ok=True);work.mkdir(exist_ok=True)
    if (work/'reference_visual_review.json').exists():
        raise ValueError('This audit requires the embedded review, with no external review file')
    for name in ('dataset.npy','metadata.csv'):
        link_readonly_reference(w/'base1250'/name,original/name)
    for path in (w/'base1250'/'images').glob('cat_*.png'):
        link_readonly_reference(path,original/'images'/path.name)
    for stage in ('pilot','full'):
        for suffix in ('generation_state.json','selection.json','labels.jsonl'):
            name=f'{stage}_{suffix}'
            shutil.copy2(w/name,work/name)
        folder=f'reference-v2-{stage}-20261114-b4'
        for path in (w/folder).glob('batch_*.npz'):
            link_readonly_reference(path,work/folder/path.name)
    for name in ('refinement.json','reproduction.json'):
        if (w/name).exists():shutil.copy2(w/name,work/name)
    # Python script directory and cwd contain no project helper modules.
    standalone=audit/'generate_dataset.py';shutil.copy2(source/'generate_dataset.py',standalone)
    environment=os.environ.copy();environment.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,str(standalone),'export','--work-dir',str(work),
        '--base',str(original),'--output',str(output),'--total',str(args.total)],
        cwd=audit,env=environment,check=True)
    names=('dataset.npy','metadata.csv','preview.png','quality_report.md','README.md','generate_dataset.py')
    source_digests={name:sha(source/name) for name in names}
    file_results={name:source_digests[name]==sha(output/name) for name in names}
    differing=[];png_manifest=hashlib.sha256()
    for i in range(args.total):
        name=f'cat_{i:04d}.png'
        source_hash=sha(source/'images'/name)
        png_manifest.update(f'{name}:{source_hash}\n'.encode('ascii'))
        if source_hash!=sha(output/'images'/name):differing.append(name)
    result=dict(count=args.total,standalone_external_helper_files=0,external_review_file_used=False,
        cached_candidates_and_FEM_labels_reused=True,all_six_deliverable_files_byte_identical=all(file_results.values()),
        deliverable_file_results=file_results,all_pngs_byte_identical=not differing,differing_pngs=differing,
        staged_deliverable_sha256=source_digests,staged_ordered_png_manifest_sha256=png_manifest.hexdigest(),
        caveat='Fresh export/packaging audit; separate GPU spot-check covers sampled regeneration.')
    (w/'package_export_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if not all(file_results.values()) or differing:raise RuntimeError('Standalone export differs; see audit report')


if __name__=='__main__':main()
