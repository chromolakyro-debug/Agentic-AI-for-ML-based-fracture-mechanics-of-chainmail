"""Reproduce the v2 165k panel at 128 midpoint steps, without altering the 64-step run."""
from pathlib import Path
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evaluation/runs/cfm_v2_step_165000'
OUT = BASE / 'sampling_128steps'
IMAGES = BASE / 'image_128 steps'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    protected = [p for p in BASE.iterdir() if p.is_file() and p.name != 'index_128steps.html'] + list((BASE/'images').glob('*.png'))
    before = {p.relative_to(BASE).as_posix():sha(p) for p in protected}
    config = copy.deepcopy(read(BASE/'config.json'))
    config['sampling']['steps'] = 128
    config['run_id'] = 'cfm_v2_step_165000_128steps'
    config['output_directory'] = OUT.relative_to(ROOT).as_posix()
    config_path = OUT/'input_config.json'
    config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-u',str(ROOT/'evaluation/run.py'),'--config',str(config_path)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluation/verify_report.py'),str(OUT)],cwd=ROOT,check=True)
    assert sha(BASE/'initial_noise.npy') == sha(OUT/'initial_noise.npy'), 'Initial noise changed'
    assert read(BASE/'targets.json') == read(OUT/'targets.json')
    assert read(BASE/'seeds.json') == read(OUT/'seeds.json')
    payload = read(OUT/'report_data.json')
    assert len(payload['rows']) == 240
    IMAGES.mkdir(exist_ok=True)
    for r in payload['rows']:
        source = OUT/r['image']
        destination = IMAGES/source.name
        shutil.copy2(source,destination)
        assert sha(destination) == r['png_sha256']
        r['image'] = '../image_128%20steps/'+destination.name
    template = (BASE/'index_en.html').read_text(encoding='utf-8')
    pattern = r'(<script id="report-data" type="application/json">).*?(</script>)'
    html, count = re.subn(pattern,lambda m:m[1]+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+m[2],template,flags=re.S)
    assert count == 1
    html = html.replace('<head>','<head><base href="sampling_128steps/">',1)
    html = re.sub(r'<title>.*?</title>','<title>CFM v2 165k — 128-step evaluation</title>',html,count=1)
    html = html.replace('<body>','<body><div class="warning"><strong>128-step sampling experiment</strong><br>Same CFM v2 165k checkpoint, 24 targets and 10 seeds. Midpoint sampling: 128 steps / 256 network evaluations. FEM labels, best-of-10 selections and retrievals are recomputed for these images. <a href="../index_en.html">Open the original 64-step report</a></div>',1)
    (BASE/'index_128steps.html').write_text(html,encoding='utf-8')
    assert all(sha(BASE/p)==digest for p,digest in before.items()), 'Original 64-step files changed'
    audit = dict(status='passed',images=240,valid_fem=sum(r['status']=='ok' for r in payload['rows']),
                 steps=128,network_evaluations=256,original_files_unchanged=before,
                 initial_noise_identical=True,targets_identical=True,seeds_identical=True,
                 image_directory=str(IMAGES),html=str(BASE/'index_128steps.html'))
    (OUT/'delivery_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='original_files_unchanged'},indent=2),flush=True)

if __name__ == '__main__':
    main()
