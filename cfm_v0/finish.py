"""Bounded completion runner; waits for training, then executes the frozen final evaluation."""
import argparse
import json
import subprocess
import sys
import time
from common import HERE, OUT, ROOT, read_json, save_json, sha_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    started=time.perf_counter()
    while not (OUT/'training_summary.json').exists():
        if not args.wait:raise RuntimeError('Main training has not completed')
        if time.perf_counter()-started>3600:raise TimeoutError('Bounded 60-minute training wait exhausted')
        time.sleep(10)
    source_files=sorted(HERE.glob('*.py'))+[ROOT/'mechanics.py',ROOT/'evaluate_mechanics.py',ROOT/'mechanics_selection.py',ROOT/'reference_style_selection.py']
    fingerprint_name='evaluation_code_sha256_at_resume.json' if (OUT/'evaluation_code_sha256.json').exists() else 'evaluation_code_sha256.json'
    save_json(OUT/fingerprint_name,{str(path.relative_to(ROOT)):sha_file(path) for path in source_files})
    completed=read_json(OUT/'completion_commands.json') if (OUT/'completion_commands.json').exists() else []
    def run(script,*arguments,artifact=None):
        tick=time.perf_counter()
        command=[sys.executable,str(HERE/script),*arguments]
        if artifact is not None and (OUT/artifact).exists() and any(row['command']==command and row['returncode']==0 for row in completed):
            print('RESUME: completed output retained',artifact,flush=True)
            return
        print('RUN',command,flush=True)
        result=subprocess.run(command,cwd=ROOT,check=True)
        completed.append(dict(command=command,seconds=time.perf_counter()-tick,returncode=result.returncode))
        save_json(OUT/'completion_commands.json',completed)
    best=read_json(OUT/'training_summary.json')['best_step']
    run('generate.py','--panel','development',artifact=f'development_{best:06d}/manifest.json')
    run('fem.py','evaluate','--folder',f'development_{best:06d}','--workers','8',artifact=f'development_{best:06d}/fem_summary.json')
    run('generate.py','--panel','final',artifact='final/manifest.json')
    run('fem.py','evaluate','--folder','final','--workers','8',artifact='final/fem_summary.json')
    run('metrics.py','mechanics',artifact='final/metrics.json')
    run('metrics.py','test',artifact='test_once.json')
    run('diagnostics.py','neighbors',artifact='final/image_neighbors.json')
    run('plots.py','review','--folder','final',artifact='final/visual_review_manifest.json')
    figure_root=HERE if OUT==(HERE/'run').resolve() else OUT
    run('plots.py','final',artifact=figure_root/'generated_examples.png')
    run('check_core.py',artifact='core_checks.json')
    save_json(OUT/'evaluation_completed.json',dict(last_completion_invocation_seconds=time.perf_counter()-started,
              sum_successful_command_seconds=sum(row['seconds'] for row in completed),
              timing_note='Sum of successful command durations; excludes failed diagnostic attempts and manual report work.',
              final_samples=640,requires_actual_visual_review=not (OUT/'final'/'visual_inspection.json').exists()))
    print('Frozen evaluation completed; visual inspection/report still required.',flush=True)


if __name__=='__main__':main()
