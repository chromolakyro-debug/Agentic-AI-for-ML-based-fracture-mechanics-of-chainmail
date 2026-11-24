"""Isolated headless Chromium test of local report JS; no user browser profile."""
import subprocess,json,re
from pathlib import Path
from utils import HERE,read,save

CHECK='''<script>
window.addEventListener('load',()=>{try{
 const sections=document.querySelectorAll('section'),cards=document.querySelectorAll('.card'),imgs=[...document.images];
 const button=document.querySelector('.buttons button'),card=button.closest('.card'),im=card.querySelector('img');
 const before=card.querySelector('.metric').textContent;
 button.click();const rotated=im.style.transform.includes('-90deg');
 card.querySelectorAll('button')[1].click();const mirrored=im.style.transform.includes('scaleX(-1)');
 const unchanged=before===card.querySelector('.metric').textContent;
 card.querySelectorAll('button')[2].click();const reset=im.style.transform==='scaleX(1) rotate(0deg)';
 const select=card.querySelector('select');select.value='yes';select.dispatchEvent(new Event('change'));
 const rated=Object.values(ratings).includes('yes');
 document.body.dataset.uiAudit=JSON.stringify({panels:sections.length,cards:cards.length,images:imgs.length,images_loaded:imgs.every(i=>i.complete&&i.naturalWidth===224),rotated,mirrored,unchanged,reset,rated,passed:rotated&&mirrored&&unchanged&&reset&&rated&&imgs.every(i=>i.complete&&i.naturalWidth===224)});
 }catch(e){document.body.dataset.uiAudit=JSON.stringify({passed:false,error:String(e)})}});
</script>'''

def main(formal=False):
    browser=Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
    if not browser.exists():raise FileNotFoundError('Install Chromium or adapt browser executable for local HTML tests')
    results=[]
    for tag in ['old_20k','old_125k']:
        suffix='' if formal else '_diagnostic'
        out=HERE/'runs'/f'{tag}{suffix}';test=HERE/'validation'/f'{tag}{suffix}_html_test.html'
        source=(out/'index.html').read_text(encoding='utf-8').replace('<head>','<head><base href="'+out.as_uri()+'/">').replace('</body>',CHECK+'</body>')
        test.write_text(source,encoding='utf-8');profile=HERE/'cache'/'headless_validation_profile'
        args=[str(browser),'--headless=new','--no-first-run','--no-default-browser-check','--disable-gpu','--disable-extensions',f'--user-data-dir={profile}',
            '--allow-file-access-from-files','--window-size=1800,1300',f'--screenshot={HERE/"validation"/(tag+suffix+"_html.png")}','--dump-dom','--virtual-time-budget=3000',test.as_uri()]
        r=subprocess.run(args,capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
        dom=r.stdout.decode('utf-8',errors='replace');(HERE/'validation'/f'{tag}_browser_stderr.txt').write_bytes(r.stderr)
        import html
        match=re.search(r'data-ui-audit="([^"]+)"',dom)
        if not match:raise RuntimeError(f'Browser did not complete report test; exit={r.returncode}, stderr={r.stderr[-2000:]!r}')
        result=json.loads(html.unescape(match.group(1)));result.update(run=tag,browser=str(browser));results.append(result)
        expected=24 if formal else 4
        assert result['passed'] and result['panels']==expected and result['cards']==expected*30
    save(HERE/('validation/formal_html_browser_checks.json' if formal else 'validation/html_browser_checks.json'),results);print(json.dumps(results,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--formal',action='store_true');a=p.parse_args();main(a.formal)
