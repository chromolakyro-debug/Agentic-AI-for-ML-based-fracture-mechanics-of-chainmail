"""Independent brute-force retrieval and existing HTML consistency verification."""
import sys
import numpy as np
from common import ROOT, RUN, read, save
sys.path.insert(0, str(ROOT / 'evaluation'))
from adapters.dataset import NumpyMetadataDataset
from verify_report import verify


def main():
    final = read(RUN / 'final_evaluation.json'); out = ROOT / final['output']
    cfg = read(out / 'config.json'); ds = NumpyMetadataDataset(cfg['dataset'], ROOT)
    rows = read(out / 'results.json'); seeds = read(out / 'seeds.json')
    train = set(ds.train); scale = np.asarray(cfg['retrieval']['distance']['scales'])
    count = 0
    for kind in ['target_retrieval', 'fem_retrieval']:
        for q in read(out / (kind + '.json')):
            query = np.asarray([q['query_Ex'], q['query_Ey']]); best = {}
            for idx in ds.train:
                e = ds.moduli[idx]
                d0 = float(np.linalg.norm((query - e) / scale)); d1 = float(np.linalg.norm((query - e[::-1]) / scale))
                item = (min(d0, d1), int(idx), d1 < d0); key = ds.source_keys[idx]
                if key not in best or item[:2] < best[key][:2]: best[key] = item
            oracle = sorted(best.values())[:10]
            assert len(q['neighbors']) == 10
            for actual, expected in zip(q['neighbors'], oracle):
                assert actual['dataset_index'] in train
                assert actual['dataset_index'] == expected[1] and actual['axis_swapped'] == expected[2]
                assert np.isclose(actual['distance'], expected[0], rtol=1e-14, atol=0)
                count += 1
            if kind == 'fem_retrieval':
                sample = min([r for r in rows if r['target_id'] == q['target_id'] and r['status'] == 'ok'], key=lambda r: (r['d_mech'], seeds.index(r['seed'])))
                assert sample['sample_key'] == q['query_sample_key']
                assert sample['Ex_FEM'] == q['query_Ex'] and sample['Ey_FEM'] == q['query_Ey']
    result = dict(status='passed', neighbor_positions=count, training_only=True, source_dedup=True,
                  axis_swap_oracle=True, best_FEM_query_selection=True, html=verify(out))
    save(RUN / 'final_retrieval_html_audit.json', result)
    print(result)


if __name__ == '__main__': main()
