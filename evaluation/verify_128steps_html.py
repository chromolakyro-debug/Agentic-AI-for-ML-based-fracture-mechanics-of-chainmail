"""Check the delivered English page in an isolated headless browser."""
from pathlib import Path
import html
import json
import re
import subprocess
from urllib.parse import unquote
from validate_html import CHECK

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evaluation/runs/cfm_v2_step_165000'
OUT=BASE/'sampling_128steps'

def main():
    source=(BASE/'index_128steps.html').read_text(encoding='utf-8')
    data=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',source,re.S)[1])
    assert '<html lang="en">' in source and '<base href="sampling_128steps/">' in source
    original=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    for row,saved in zip(data['rows'],original):
        assert {k:v for k,v in row.items() if k!='image'}=={k:v for k,v in saved.items() if k!='image'}
        assert (OUT/unquote(row['image'])).resolve().parent==BASE/'image_128 steps'
        assert (OUT/unquote(row['image'])).read_bytes()==(OUT/saved['image']).read_bytes()
    for kind in ['target_retrieval','fem_retrieval']:
        for q in data[kind]:
            for n in q['neighbors']:assert (OUT/n['display_image']).exists()
    for target in data['targets']:
        best=min((r for r in data['rows'] if r['target_id']==target['target_id'] and r['status']=='ok'),key=lambda r:(r['d_mech'],r['seed']))
        query=next(q for q in data['fem_retrieval'] if q['target_id']==target['target_id'])
        assert query['query_sample_key']==best['sample_key']
    for href in re.findall(r'href="([^"]+)"',source):
        if href=='sampling_128steps/':continue
        assert (OUT/unquote(href)).exists(),href
    test=OUT/'browser_validation.html'
    test.write_text(source.replace('<base href="sampling_128steps/">','<base href="'+OUT.as_uri()+'/">').replace('</body>',CHECK+'</body>'),encoding='utf-8')
    browser='C:/Program Files/Google/Chrome/Application/chrome.exe'
    args=[browser,'--headless=new','--no-first-run','--no-default-browser-check','--disable-gpu','--disable-extensions',
          '--user-data-dir='+str(OUT/'browser_profile'),'--allow-file-access-from-files','--window-size=1800,1300',
          '--screenshot='+str(OUT/'browser_preview.png'),'--dump-dom','--virtual-time-budget=5000',test.as_uri()]
    result=subprocess.run(args,capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    dom=result.stdout.decode('utf-8',errors='replace')
    match=re.search(r'data-ui-audit="([^"]+)"',dom)
    assert match,'Browser audit missing'
    audit=json.loads(html.unescape(match[1]))
    assert audit['passed'] and audit['panels']==24 and audit['cards']==720,audit
    audit.update(delivered_paths_verified=True,best_samples_verified=True,labels_match_128step_results=True)
    (OUT/'html_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
