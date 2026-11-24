"""Check the published Lab 4 metrics using only the standard library."""
import csv, math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evaluation/runs/cfm_v2_step_165000'
for sub,expected in [('',(109,65)),('sampling_128steps',(106,69))]:
    with (base/sub/'results.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==240
    assert len({(r['target_id'],r['seed']) for r in rows})==240
    assert all(r['status']=='ok' for r in rows)
    s_hits=sum(abs(float(r['delta_s']))<=math.log(1.05) and abs(float(r['delta_a']))<=math.log(1.10) for r in rows)
    e_hits=sum(abs(float(r['Ex_FEM'])/float(r['Ex_target'])-1)<=.05 and abs(float(r['Ey_FEM'])/float(r['Ey_target'])-1)<=.05 for r in rows)
    assert (s_hits,e_hits)==expected, (sub,s_hits,e_hits)
    print(f'{sub or "64 steps"}: 240 valid; s-a hits {s_hits}; Ex/Ey hits {e_hits}')
print('Published results checks passed.')
