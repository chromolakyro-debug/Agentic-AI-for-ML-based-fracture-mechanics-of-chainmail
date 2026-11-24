"""Bind completed audits into the reproducible package; never publish live data."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import reference_style_workflow as flow


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--total',type=int,default=8000)
    args=p.parse_args();w=args.work_dir;release=w/'release';root=Path(__file__).resolve().parent
    status=flow.read_json(w/'finishing_state.json',{})
    if status.get('stage')!='staging_ready_for_final_visual_review':
        raise ValueError('The single-writer finishing job must be complete before finalizing review metadata')
    data=flow.read_json(w/'reference_visual_review.json')
    corrections=flow.read_json(w/'pending_review_reason_corrections.json',[])
    for correction in corrections:
        record=data['source_review_pages'][correction['page_identity']]
        cell=correction['cell'];h=correction['pixels_sha256']
        if record['ordered_pixel_hashes'][cell-1]!=h or h not in data['excluded_hashes']:
            raise ValueError('Reason-only correction is not bound to the same excluded pixels')
        old=record['excluded_cells'][str(cell)];new=correction['reason']
        if old!=new:
            record['excluded_cells'][str(cell)]=new;data['excluded_hashes'][h]=new
            data.setdefault('reason_only_corrections',[]).append(dict(correction,previous_reason=old))
    baseline=flow.read_json(w/'base_pair_audit.json')
    baseline['base_identity']=flow.BASE_IDENTITY
    data['fixed_base_pair_audit']=baseline
    flow.write_json(w/'reference_visual_review.json',data)
    with (release/'metadata.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    flow.write_report(SimpleNamespace(work_dir=w,output=release),rows,flow.read_json(w/'final_pair_audit.json'))
    subprocess.run([sys.executable,str(root/'assemble_reference_generator.py'),str(release/'generate_dataset.py'),
                    '--work-dir',str(w)],cwd=root,check=True)
    subprocess.run([sys.executable,str(root/'check_reference_package_export.py'),'--work-dir',str(w),
                    '--total',str(args.total)],cwd=root,check=True)
    flow.write_json(w/'finalized_staging.json',dict(count=len(rows),review_corrections_applied=True,
        whole_dataset_near_duplicate_audit_included=True,standalone_export_audit='pass',
        package_audit_sha256=hashlib.sha256((w/'package_export_check.json').read_bytes()).hexdigest(),
        publication_performed=False))
    print('Finalized staging package; live dataset still untouched.',flush=True)


if __name__=='__main__':main()
