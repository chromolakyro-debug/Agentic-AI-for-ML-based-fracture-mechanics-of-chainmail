"""Post-training, human-reviewed report and self-contained numerical audit bundle.

Run only after workflow completion and fixed-seed visual inspection. Does not
change training state or choose checkpoints using FEM results.
"""
import difflib
import json
import shutil
import zipfile
import numpy as np
from common import ROOT, HERE, RUN, read, save, sha


def main():
    assert read(RUN / 'workflow_complete.json')['status'] == 'complete'
    assert read(RUN / 'final_retrieval_html_audit.json')['status'] == 'passed'
    cfg = read(RUN / 'config.json')
    summary = read(RUN / 'training_summary.json')
    final = read(RUN / 'final_evaluation.json')
    out = ROOT / final['output']
    assert sha(ROOT / cfg['continuation']['source_checkpoint']) == cfg['continuation']['sha256']
    note = RUN / f"visual_observation_{summary['best_step']:06d}.md"
    assert note.exists(), 'Inspect the final fixed-seed sheet and record observations first.'
    source_rows = read(ROOT / 'evaluation/runs/cfm_v1_step_280000/results.json')
    final_rows = read(out / 'results.json')
    def metrics(rows):
        valid = [r for r in rows if r['status'] == 'ok']
        return dict(valid=len(valid), failed=len(rows)-len(valid),
                    mean_d=float(np.mean([r['d_mech'] for r in valid])),
                    hits_sa=sum(r['hit_s_a'] for r in valid),
                    hits_exey=sum(r['hit_Ex_Ey'] for r in valid))
    history = [dict(step=280000, validation_mse=0.09608178820849678, **metrics(source_rows))]
    seen = {280000}
    numerical = RUN / 'periodic_numerical_results'
    for event in read(RUN / 'evaluations.json'):
        step = event['checkpoint_step']
        if step in seen:
            continue
        seen.add(step)
        folder = ROOT / event['output']
        history.append(dict(step=step, validation_mse=event['validation_mse'], **metrics(read(folder / 'results.json'))))
        dest = numerical / f'step_{step:06d}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['results.json', 'results.csv', 'summary.json', 'manifest.json', 'config.json', 'targets.json', 'seeds.json']:
            p = folder / name
            if p.exists(): shutil.copyfile(p, dest / name)
    save(RUN / 'comparison_with_parent.json', history)
    # Keep independent baseline data for checking paired summary claims.
    save(RUN / 'parent_final_results.json', source_rows)
    differences = []
    for name in ['train.py', 'control.py', 'augmentation.py']:
        a = (ROOT / 'cfm_v1' / name).read_text(encoding='utf-8').splitlines(keepends=True)
        b = (HERE / name).read_text(encoding='utf-8').splitlines(keepends=True)
        differences.extend(difflib.unified_diff(a, b, fromfile='cfm_v1/'+name, tofile='cfm_v1_cont1/'+name))
    (RUN / 'training_code_changes.diff').write_text(''.join(differences), encoding='utf-8')
    report_path = HERE / 'REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    report = report.replace('每20k进行完整240样本FEM评估并留存完整checkpoint',
                            '每20k检查并评估当时的best-validation，完全相同的checkpoint/config复用缓存；本轮7个不同checkpoint各完成240样本评估，并每20k留存完整训练checkpoint')
    old = '视觉质量不能由MSE或FEM替代，待固定样本实际查看后补充观察；没有宣称memorization或多样性已通过量化检验。'
    visual = note.read_text(encoding='utf-8').strip()
    if old in report:
        report = report.replace(old, visual + ' 固定样本为全部24个targets的第一枚预定seed730000，未经质量筛选；这些观察不等同于全部240图的人工评分，没有完成memorization或多样性的量化检验。')
    marker = '\n## 本轮核验补充\n'
    report = report.split(marker)[0] + marker
    extra_restart = summary['restart_count'] - read(RUN / 'parent_training_summary.json')['restart_count'] if (RUN / 'parent_training_summary.json').exists() else sum(280000 < s <= summary['step'] for s in range(20001, summary['step']+1, 20000))
    report += f"\n本轮实际新增LR restart {extra_restart}次；累计restart {summary['restart_count']}次。原v1 checkpoint的SHA256再次核验未变。最终HTML数值、480个retrieval位置的train-only去重/轴交换/排序和FEM query选择已独立核对通过。\n\n"
    report += '| Best-validation checkpoint | validation MSE | mean(d_mech) | s/a联合命中 | Ex/Ey联合命中 |\n|---|---:|---:|---:|---:|\n'
    for r in history:
        report += f"| {r['step']//1000}k | {r['validation_mse']:.9f} | {r['mean_d']:.6f} | {r['hits_sa']}/240 | {r['hits_exey']}/240 |\n"
    report += '\nMSE改善与机械指标并非单调一致，最终模型始终按validation MSE选择，未按此表中的FEM表现挑选。逐周期停止判定见run/plateau_history.json；完整原始曲线、固定seed图、逐样本FEM值及失败状态均保留。\n'
    if summary['reason'] != 'convergence_criterion':
        report += '\n本轮按用户明确确认的480k上限停止；最后完整周期的validation改善仍为0.395%，严格平台期计数为0，未满足更严格的收敛停止条件，不能声称已经充分收敛。训练与最终评估已结束，不再自动延长。\n'
    report_path.write_text(report, encoding='utf-8')
    (HERE / 'START_HERE.md').write_text(
        '# CFM v1 continuation 1 审阅入口\n\n先读REPORT.md、training_curves.png和run/comparison_with_parent.json。'
        'final_evaluation/index.html显示最终240张生成图及两组train-only机械retrieval。'
        'run/fixed_seed_*.png是固定seed730000、全部24个targets的未经筛选对照，visual_observation文件记录实际人工查看。\n\n'
        'run/training.jsonl前280k为父v1原始历史，其后为本次续训；完整状态继承，只有停止门槛和输出/保留规则改变。'
        'run/config.json、parent_config.json、training_code_changes.diff说明差异；checks.json为恢复验证，verification.json与final_retrieval_html_audit.json为最终核验。'
        'run/periodic_numerical_results和parent_final_results.json支持独立核算；final_fem_inputs.npz是最终FEM二值输入。\n\n'
        '不包含大模型权重和训练图片，checkpoint_inventory.json列保存路径和SHA256。'
        'README.md和run/commands.jsonl记录恢复/复现命令。此panel复用此前targets/seeds，不是新的holdout；不据此宣称消除memorization。\n', encoding='utf-8')
    # Include pipeline source, frozen configs, compact numerical records and final
    # review page; omit all checkpoint tensors and raw floating generation arrays.
    files = {}
    def add(path, archive): files[archive] = path
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in {'.py', '.md', '.png'}: add(p, p.name)
    for base in [RUN, HERE / 'configs']:
        for p in base.rglob('*'):
            if p.is_file() and p.suffix not in {'.pt', '.tmp', '.pyc', '.zip'} and '__pycache__' not in p.parts and p.name != 'audit_package.json':
                add(p, p.relative_to(HERE).as_posix())
    for p in out.rglob('*'):
        if p.is_file() and 'arrays' not in p.relative_to(out).parts and p.name != 'initial_noise.npy':
            add(p, 'final_evaluation/' + p.relative_to(out).as_posix())
    for p in (ROOT / 'evaluation').rglob('*.py'):
        if not {'cache', 'runs', '__pycache__'}.intersection(p.relative_to(ROOT / 'evaluation').parts):
            add(p, 'evaluation_source/' + p.relative_to(ROOT / 'evaluation').as_posix())
    manifest = dict(files=[dict(path=n, bytes=p.stat().st_size, sha256=sha(p)) for n,p in sorted(files.items())])
    save(HERE / 'AUDIT_FILES.json', manifest)
    add(HERE / 'AUDIT_FILES.json', 'AUDIT_FILES.json')
    package = HERE / 'cfm_v1_cont1_ai_review.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        for n,p in sorted(files.items()): z.write(p,n)
    import hashlib
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for entry in manifest['files']:
            assert hashlib.sha256(z.read(entry['path'])).hexdigest() == entry['sha256']
        assert len(json.loads(z.read('final_evaluation/results.json'))) == 240
    save(RUN / 'audit_package.json', dict(path=str(package), bytes=package.stat().st_size, sha256=sha(package), every_member_hash_verified=True))
    print(json.dumps(dict(status='passed', archive=str(package), megabytes=package.stat().st_size/1e6, history=history), indent=2))


if __name__ == '__main__':
    main()
