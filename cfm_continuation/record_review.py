"""Bind explicitly supplied observations from actual pixel inspection to image hashes."""
import paths
import argparse
from collections import Counter
from datetime import datetime,timezone
from common import OUT, read_json,save_json,sha_file


def record(step,final=False):
    folder=OUT/'final' if final else OUT/f'development_{step:06d}'
    supplied=read_json(paths.HERE/'reviews'/('final.json' if final else f'{step:06d}.json'))
    manifest=read_json(folder/'manifest.json')
    source_samples=[r for r in manifest['samples'] if r['seed'] in manifest['seeds'][:2]]
    assert len(supplied['observations'])==len(source_samples)==(40 if final else 30)
    samples=[]
    for item,row in zip(supplied['observations'],source_samples):
        assert item[0] in (0,1,2)
        samples.append(dict(key=row['key'],sha256_pixels=row['sha256_pixels'],rating=item[0],note=item[1]))
    save_json(folder/'visual_review.json',dict(reviewer='Assistant actual pixel inspection, subjective AI review; NOT human confirmation',
        rubric={'2':'Recognizable coherent cat (portrait or whole body), intelligible pose; minor stylization/artifacts allowed',
                '1':'Cat cues present but conspicuous anatomical distortion, fragmentation or unintended truncation',
                '0':'Not reliably recognizable as a cat / severe disconnected anatomy'},
        scope='First 2 frozen seeds for every target; no appearance or FEM selection; '+('40 historical final benchmark images' if final else 'all 30 development images'),
        step=step,recorded_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha_file(folder/'manifest.json'),
        counts=dict(Counter(str(r['rating']) for r in samples)),samples=samples,
        summary=supplied['summary'],change_from_previous=supplied['change_from_previous'],
        meaningful_visual_improvement=supplied['meaningful_visual_improvement'],decision=supplied['decision']))
    print(dict(Counter(str(r['rating']) for r in samples)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);p.add_argument('--final',action='store_true');a=p.parse_args();record(a.step,a.final)
