"""User-authorized, audited budget-only migration after the original workflow exits."""
import copy
import datetime
import shutil
import torch
from common import ROOT, RUN, read, save, sha


def main():
    authorization = read(RUN / 'budget_extension_authorization.json')
    summary = read(RUN / 'training_summary.json')
    assert (RUN / 'workflow_complete.json').exists(), 'Wait for the original workflow to exit first'
    assert summary['step'] == authorization['review_at_step'] == 200000
    assert summary['reason'] == 'update_budget'
    history = read(RUN / 'plateau_history.json')
    latest = history[-1]
    assert latest['step'] == 200000 and not latest['stop']
    assert latest['relative_improvement'] > 0, 'No continuing cycle-best validation improvement; do not extend'
    assert not (RUN / 'budget_extension_applied.json').exists()
    cfg = read(RUN / 'config.json')
    extended = copy.deepcopy(cfg)
    extended['limits']['max_updates'] = authorization['proposed_max_updates']
    extended['limits']['max_training_seconds'] = authorization['proposed_max_training_seconds']
    assert all(cfg[k] == extended[k] for k in cfg if k != 'limits')
    source = RUN / 'checkpoints/last.pt'
    retained = RUN / 'checkpoints/step_200000.pt'
    assert retained.exists() and sha(source) == sha(retained), 'Preserve the original complete 200k checkpoint first'
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert state['step'] == 200000 and state['config'] == cfg
    archived = RUN / 'budget_boundary_200000'
    archived.mkdir(exist_ok=False)
    names = ['config.json', 'frozen_config.json', 'training_summary.json', 'workflow_complete.json',
             'final_evaluation.json', 'final_metrics.json', 'audit_package.json', 'checkpoint_inventory.json']
    for name in names:
        p = RUN / name
        if p.exists():
            shutil.copyfile(p, archived / name)
    for name in ['REPORT.md', 'START_HERE.md']:
        p = ROOT / 'cfm_v1' / name
        if p.exists():
            shutil.copyfile(p, archived / name)
    before_hash = sha(source)
    state['config'] = extended  # Only limits metadata changes; every training-state object is untouched.
    temporary = source.with_suffix('.budget.tmp')
    torch.save(state, temporary)
    restored = torch.load(temporary, map_location='cpu', weights_only=False)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            import numpy as np
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b)
            else:
                assert a == b

    original = torch.load(retained, map_location='cpu', weights_only=False)
    for key in original:
        if key != 'config':
            equal(original[key], restored[key])
    save(RUN / 'config.json', extended)
    frozen = read(RUN / 'frozen_config.json')
    frozen['sha256'] = sha(RUN / 'config.json')
    save(RUN / 'frozen_config.json', frozen)
    temporary.replace(source)
    save(RUN / 'budget_extension_applied.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), at_step=200000,
        old_limits=cfg['limits'], new_limits=extended['limits'],
        cycle_relative_improvement=latest['relative_improvement'],
        original_checkpoint_sha256=before_hash, migrated_checkpoint_sha256=sha(source),
        all_non_config_state_exactly_equal=True, changed_fields=['config.limits'],
        reason='User authorized extending budget while validation continues improving; plateau and LR untouched'))
    # Move completion markers so workflow resumes rather than finalizes immediately.
    # Their exact original contents were already archived above.
    (RUN / 'training_summary.json').replace(archived / 'training_summary.json')
    (RUN / 'workflow_complete.json').replace(archived / 'workflow_complete.json')
    print('Budget extended; resume workflow from unchanged step-200000 training state.')


if __name__ == '__main__':
    main()
