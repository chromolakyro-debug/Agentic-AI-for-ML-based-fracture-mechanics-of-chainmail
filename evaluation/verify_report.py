import json,re
from pathlib import Path
import numpy as np
from utils import read,sha,distance

def verify(out):
    out=Path(out);text=(out/'index.html').read_text(encoding='utf-8')
    embedded=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',text,re.S).group(1))
    assert embedded==read(out/'report_data.json')
    assert embedded['rows']==read(out/'results.json')
    assert embedded['summary']==read(out/'summary.json')
    # The table source is embedded verbatim; formatted labels are derived from this one object in the browser.
    for kind in ['target_retrieval','fem_retrieval']:
        saved=read(out/f'{kind}.json')
        for q,original in zip(embedded[kind],saved):
            assert {k:v for k,v in q.items() if k!='neighbors'}=={k:v for k,v in original.items() if k!='neighbors'}
            keys=[];dist=[]
            for r,orig in zip(q['neighbors'],original['neighbors']):
                assert {k:v for k,v in r.items() if k!='display_image'}==orig
                assert sha(out/r['display_image'])==r['png_sha256'];keys.append(r['source_key']);dist.append(r['distance'])
                assert np.isclose(distance([q['query_Ex'],q['query_Ey']],[r['Ex_matched'],r['Ey_matched']],embedded['distance']),r['distance'],rtol=1e-14,atol=0)
            assert len(keys)==len(set(keys));assert dist==sorted(dist)
    return dict(status='passed',rows=len(embedded['rows']),targets=len(embedded['targets']),checks=['embedded JSON equals result tables','retrieval PNG hashes','distance recomputed','source deduplication','rank order'])

if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
