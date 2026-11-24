"""Freeze a single-factor ablation against the actual, completed CFM v1 run."""
import copy
import csv
import shutil
import numpy as np
from PIL import Image
from common import ROOT, RUN, DATA, read, save, sha
from audit_verify import verify


def main():
    if (RUN / 'config.json').exists():
        raise RuntimeError('Configuration already frozen; use workflow.py to resume')
    reference = ROOT / 'cfm_v1/run'
    cfg = read(reference / 'config.json')
    assert sha(reference / 'config.json') == read(reference / 'frozen_config.json')['sha256']
    assert read(reference / 'workflow_complete.json')['status'] == 'complete'
    for key, path in dict(metadata=DATA/'metadata.csv', array=DATA/'dataset.npy',
                          data=reference/'data.npz', splits=reference/'splits.json',
                          model_code=ROOT/'cfm_v0/model.py').items():
        assert sha(path) == cfg['input_hashes'][key], key
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ['data.npz', 'splits.json', 'data_audit.json']:
        shutil.copyfile(reference/name, RUN/name)
    # Verify current pixels and metadata, rather than treating a past audit as current evidence.
    rows = list(csv.DictReader((DATA/'metadata.csv').open(encoding='utf-8-sig', newline='')))
    images = np.load(DATA/'dataset.npy', mmap_mode='r')
    d = np.load(RUN/'data.npz', allow_pickle=False)
    assert images.shape == (7857, 224, 224) and images.dtype == np.uint8
    for i, row in enumerate(rows):
        assert row['new_id'] == d['sample_ids'][i]
        assert row['original_id'] == d['original_ids'][i]
        assert int(row['curation.dataset_index']) == i
        assert [float(row['Ex_app']), float(row['Ey_app'])] == d['moduli'][i].tolist()
        with Image.open(DATA/'images'/row['filename']) as image:
            pixels = np.asarray(image.convert('L'))
        assert np.isin(pixels, [0, 255]).all() and np.array_equal(pixels//255, images[i])
    modified = copy.deepcopy(cfg)
    modified['version'] = 'cfm_v2'
    modified['augmentation'] = 'none; original training images and physical Ex/Ey unchanged; unused D4 RNG draws retained for pairing with CFM v1'
    modified['sampling']['order'] = 'draw original train row with replacement; no augmentation; no class equalization; same original-row weights as CFM v1'
    # Exactly three semantic/config-label differences; all training controls match final v1.
    expected = copy.deepcopy(modified)
    expected['version'] = cfg['version']
    expected['augmentation'] = cfg['augmentation']
    expected['sampling']['order'] = cfg['sampling']['order']
    assert expected == cfg
    save(RUN/'config.json', modified)
    save(RUN/'frozen_config.json', dict(sha256=sha(RUN/'config.json')))
    ref = RUN/'reference_v1'; ref.mkdir(exist_ok=True)
    for name in ['config.json', 'frozen_config.json', 'training_summary.json', 'validation_history.csv',
                 'plateau_history.json', 'final_evaluation.json', 'final_metrics.json', 'checks.json']:
        shutil.copyfile(reference/name, ref/name)
    shutil.copyfile(ROOT/'cfm_v1/train.py', ref/'train.py')
    shutil.copyfile(ROOT/'cfm_v1/REPORT.md', ref/'REPORT.md')
    template = read(ROOT/read(reference/'final_evaluation.json')['config'])
    save(RUN/'evaluation_template.json', template)
    dependencies = [ROOT/template[k] for k in ['targets','diagnostic_targets','grid','seeds']]
    dependencies += list((ROOT/'evaluation').glob('*.py')) + list((ROOT/'evaluation/adapters').glob('*.py'))
    dependencies += [ROOT/'mechanics.py', ROOT/'cfm_v0/model.py']
    save(RUN/'reference_dependencies.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies})
    save(RUN/'ablation_protocol.json', dict(
        reference_thread=dict(title='CFM v0-v1', id='01a1ddda-30fc-72f0-9115-a1a1823265f7'),
        reference_config_sha256=sha(reference/'config.json'),
        reference_train_code_sha256=sha(ROOT/'cfm_v1/train.py'),
        changed_fields=['version', 'augmentation', 'sampling.order'],
        intervention='Disable training D4 pixel transformations and Ex/Ey swaps only',
        random_pairing='Same initialization seed and RNG draws, including discarded augmentation draws; preserve sample/noise/time streams. cuDNN benchmark remains enabled as in v1; bitwise historical replay is not claimed.',
        budget='Use final v1 maximum 300000 updates / 64800 active seconds from the outset; same plateau rule can select a different stopping step.',
        comparisons='Same-step validation curves and shared evaluation targets/seeds; separately report each run selected by identical early-stopping policy. Single paired training seed, not multi-seed statistical evidence.',
        checkpoint_retention='First adjacent validation absolute relative change <1% activates retention; save only at global step multiples of 20000 at or after activation. last every 1000; best by fixed EMA validation only',
        current_png_array_metadata_check=dict(status='passed', count=len(rows)),
        untouched_test=True))
    save(RUN/'independent_data_audit.json', verify(RUN.parent))
    print('PASS: all 7857 image/ID/label links; identical splits, weights, normalization and controls; only training augmentation disabled.')


if __name__ == '__main__':
    main()
