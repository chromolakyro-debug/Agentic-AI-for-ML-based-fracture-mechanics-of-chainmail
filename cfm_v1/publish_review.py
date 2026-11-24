"""Final measured synthesis and portable audit, including authorized budget revisions."""
import json
import shutil
import zipfile
import numpy as np
from common import HERE, ROOT, RUN, read, save, sha
from audit_verify import verify


def main():
    assert (RUN / 'workflow_complete.json').exists()
    cfg = read(RUN / 'config.json'); end = read(RUN / 'training_summary.json')
    final = read(RUN / 'final_evaluation.json'); out = ROOT / final['output']
    result = verify(HERE, complete=True)
    save(RUN / 'independent_complete_audit.json', result)
    rows = read(out / 'results.json'); valid = [r for r in rows if r['status'] == 'ok']
    targets = read(out / 'targets.json')['targets']; grid = read(out / 'grid.json')
    retrieval = read(out / 'target_retrieval.json')
    stats = dict(mean_d=float(np.mean([r['d_mech'] for r in valid])),
                 retrieval_top1_mean_d=float(np.mean([r['neighbors'][0]['distance'] for r in retrieval])),
                 hits_sa=sum(r['hit_s_a'] for r in valid), hits_exey=sum(r['hit_Ex_Ey'] for r in valid))
    for axis in ['s', 'a', 'Ex', 'Ey']:
        e = np.array([r['delta_' + axis] for r in valid])
        stats[axis] = dict(median_abs=float(np.median(abs(e))), p90_abs=float(np.quantile(abs(e), .9)), bias=float(e.mean()))
    for label, check in [('extreme_a', lambda r: abs(r['a_target']) > .8), ('middle_a', lambda r: abs(r['a_target']) < .3)]:
        subset = [r for r in valid if check(r)]
        stats[label] = dict(n=len(subset), mean_d=float(np.mean([r['d_mech'] for r in subset])), hits_sa=sum(r['hit_s_a'] for r in subset))
    scans = []
    for fixed_key, varying in [('grid_row', 's'), ('grid_col', 'a')]:
        cross = 'a' if varying == 's' else 's'
        for index in sorted({t[fixed_key] for t in targets}):
            slopes = []; leaks = []
            for seed in read(out / 'seeds.json'):
                subset = [r for r in valid if r['seed'] == seed and r[fixed_key] == index]
                if len(subset) < 2: continue
                x = [r[varying + '_target'] for r in subset]
                slopes.append(float(np.polyfit(x, [r[varying + '_FEM'] for r in subset], 1)[0]))
                leaks.append(float(np.polyfit(x, [r[cross + '_FEM'] for r in subset], 1)[0]))
            scans.append(dict(fixed_axis=cross, bin_index=index, varying_axis=varying,
                              response_slope_mean=float(np.mean(slopes)), cross_slope_mean=float(np.mean(leaks)),
                              cross_slope_abs_mean=float(np.mean(np.abs(leaks))), n_seeds=len(slopes)))
    save(RUN / 'final_scan_slopes.json', scans)
    save(RUN / 'final_detailed_metrics.json', stats)
    d = np.load(RUN / 'data.npz'); sa = d['sa'][d['train']]; coverage = []
    for t in targets:
        c, r = t['grid_col'] - 1, t['grid_row'] - 1
        smask = (sa[:, 0] >= grid['s_edges'][c]) & (sa[:, 0] < grid['s_edges'][c + 1])
        amask = ((sa[:, 1] >= grid['a_edges'][r]) & (sa[:, 1] < grid['a_edges'][r + 1])) | ((-sa[:, 1] >= grid['a_edges'][r]) & (-sa[:, 1] < grid['a_edges'][r + 1]))
        coverage.append(dict(target_id=t['target_id'], train_images_either_orientation=int((smask & amask).sum())))
    save(RUN / 'training_target_bin_coverage.json', coverage)
    center = [r for r in scans if r['fixed_axis'] == 'a' and r['bin_index'] in [3, 4]]
    a_scans = [r for r in scans if r['varying_axis'] == 'a']
    milestones = []
    for e in read(RUN / 'evaluations.json'):
        if any(x['step'] == e['checkpoint_step'] for x in milestones): continue
        path = ROOT / e['output']; rr = read(path / 'results.json'); ok = [r for r in rr if r['status'] == 'ok']
        milestones.append(dict(step=e['checkpoint_step'], val_mse=e['validation_mse'], valid=len(ok),
                               mean_d=float(np.mean([r['d_mech'] for r in ok])), hits_sa=sum(r['hit_s_a'] for r in ok),
                               hits_exey=sum(r['hit_Ex_Ey'] for r in ok)))
        destination = RUN / 'milestone_evaluations' / str(e['checkpoint_step']); destination.mkdir(parents=True, exist_ok=True)
        for name in ['results.json', 'results.csv', 'summary.json', 'config.json', 'targets.json', 'seeds.json', 'manifest.json']:
            if (path / name).exists(): shutil.copyfile(path / name, destination / name)
    save(RUN / 'milestone_metrics.json', milestones)
    hist = read(RUN / 'plateau_history.json')
    windows = '、'.join(f"{r['step']//1000}k: {100*r['relative_improvement']:.3f}%" for r in hist[-4:])
    old = read(ROOT / 'evaluation/comparison/overall.json')[1]
    table = '\n'.join(f"| {r['step']:,} | {r['val_mse']:.6f} | {r['mean_d']:.4f} | {r['hits_sa']}/240 | {r['hits_exey']}/240 |" for r in milestones)
    notes = '\n\n'.join(p.read_text(encoding='utf-8').strip() for p in sorted(RUN.glob('visual_observation_*.md')))
    save(RUN / 'review_source_manifest.json', dict(visual_observation_files=[p.name for p in sorted(RUN.glob('visual_observation_*.md'))], numerical_selection='best validation only; no FEM checkpoint selection'))
    (RUN / 'visual_observations.md').write_text(notes + '\n', encoding='utf-8')
    text = f'''# CFM v1 实际训练与评估

训练至{end['step']:,} updates，累计active训练/validation/checkpoint时间{end['training_seconds']/3600:.3f}小时，{end['samples_seen']:,}次样本呈现，等效{end['equivalent_epochs']:.2f} epochs（有放回采样）。停止原因`{end['reason']}`；最佳EMA在{end['best_step']:,}，validation velocity MSE={end['best_val_mse']:.9f}。最终评估按best_val_mse选择，没有按FEM换模型。

沿用11,529,633参数Conditional U-Net、s/a conditioning、Gaussian linear flow matching、AdamW、EMA、gradient clip及batch32/bfloat16。GPU RTX4090，实际每步吞吐/显存完整保存在training.jsonl。新数据7,857张，固定train/validation/test=6285/786/786；逐图关联和整类映射检查通过。随机初始化新模型。z-score mean/std仅用新训练集重算；原数据不改。oversampling沿用原图s/a的8×12训练集分箱、逆平方根频数与均匀混合、权重截断后归一，不做类别均衡，也不对增强后的分布二次加权。

每次先抽原图，再等概率CCW旋转0/90/180/270度，独立50%左右翻转；严格先旋转后翻转。奇数90度先交换物理Ex/Ey，再计算`s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]`、`a=log(Ex/Ey)`及z-score。不插值、不反色、不生成永久副本。validation不增强，786张固定顺序、noise/time，每5k用EMA评估。MSE是像素均值`(v_theta-(x-epsilon))²`，不是FEM误差。

LR每20k updates一个cosine周期，首峰2e-4、200-step warmup，后续峰1e-4、谷3e-5、周期不变；实际restart {end['restart_count']}次，仅改变LR。80k后完整周期比较，连续2次改善<1%启动评估，连续3次<0.5%后还必须再过一个完整restart周期；有效改善重置停止计数。最近四个窗口：{windows}。用户在训练中授权延长预算；200k时仍改善0.634%，故上限由200k/12小时调整为300k/18小时。仅更新预算字段，其余恢复状态逐项完全一致，证据在budget_extension_applied.json；不是重新初始化训练。

阶段A在160k启动。所有评估复用既有24 targets、10 seeds、midpoint 64步采样、阈值0二值化，真实原FEM求解；该panel已多次使用，不能称为新holdout。validation与FEM分开记录。

| checkpoint | validation MSE | mean(d_mech) | s/a联合命中 | Ex/Ey ±5%联合命中 |
| --- | --- | --- | --- | --- |
{table}

`d_mech=sqrt((Ex_FEM-Ex_target)^2+(Ey_FEM-Ey_target)^2)`；s/a命中阈值固定为`|Δs|≤log(1.05)`且`|Δa|≤log(1.10)`。最终FEM有效{len(valid)}/240，失败{240-len(valid)}计入分母。|Δs| median/P90={stats['s']['median_abs']:.4f}/{stats['s']['p90_abs']:.4f}，|Δa|={stats['a']['median_abs']:.4f}/{stats['a']['p90_abs']:.4f}。

旧125k同panel的mean(d)={old['mean_d_mech']:.4f}、s/a命中{old['hits_s_a']}/240；本轮为{stats['mean_d']:.4f}、{stats['hits_sa']}/240。新旧数据、划分、训练量和增强均不同，不能当作augmentation单因素消融。新训练集机械retrieval top1的mean(d)={stats['retrieval_top1_mean_d']:.4f}，与随机生成分开比较；页面第三行query经过FEM best-of-10筛选，不能替代随机生成成绩。

极端各向异性`|a_target|>0.8`命中{stats['extreme_a']['hits_sa']}/{stats['extreme_a']['n']}，中部`|a_target|<0.3`命中{stats['middle_a']['hits_sa']}/{stats['middle_a']['n']}。各格训练覆盖、误差和逐seed扫描斜率见training_target_bin_coverage.json、final_scan_slopes.json；理想响应斜率1、交叉斜率0，仅作描述性诊断，不声称统计独立或物理可达性证明。

固定中部a扫描s，平均响应斜率为{center[0]['response_slope_mean']:.3f}/{center[1]['response_slope_mean']:.3f}，逐seed交叉斜率绝对值均值为{center[0]['cross_slope_abs_mean']:.3f}/{center[1]['cross_slope_abs_mean']:.3f}，总体刚度跟随较好。固定s扫描a，响应斜率范围{min(r['response_slope_mean'] for r in a_scans):.3f}–{max(r['response_slope_mean'] for r in a_scans):.3f}，最低/最高s两列存在各向异性响应不足。极端a格T01/T02分别只有{coverage[0]['train_images_either_orientation']}/{coverage[1]['train_images_either_orientation']}张训练图可经轴交换落入该格，覆盖稀疏与困难区域重合，但尚不能证明因果。

固定seed全24目标的图像已逐milestone查看，猫形与失败例并存，局部轮廓、风格会随checkpoint改变；validation下降没有对应所有条件下单调视觉改善。极端条件反复出现拉长/圆钝，几何和像素结构存在触边、局部残缺。完整观察记录见run/visual_observations.md；这里只做限定样本的实际视觉检查，没有宣称全部图人工合格。多样性与memorization留给完整HTML及训练图人工对照，机械retrieval不是视觉近邻，不能仅凭像素不相同证明创新。

现有证据确认训练后期收益减小、误差及视觉缺陷依赖条件；覆盖不足、conditioning难度与二值化各自贡献尚未被隔离验证，不把猜测当结论。本轮未扩数据、加参数或引入surrogate。下一步宜先人工审阅失败条件和训练覆盖，再决定单一受控改动。

完整结果页面：`{final['output']}/index.html`；训练曲线training_curves.png。独立审计已重算训练集归一化/权重、逐step LR、validation选择、plateau状态、FEM输入hash及距离/样本方差；两组共480个retrieval位置另用brute-force oracle核验train-only、来源去重和轴交换，HTML内嵌数据与结果表一致。用户曾要求10k留存，后改20k；150k/160k/170k保留，180k起每20k完整快照，均有SHA256。恢复命令、实际命令、配置版本和权重路径见README.md、run/commands.jsonl、run/checkpoint_inventory.json。审计包不含完整训练图或模型权重，包含原始loss/LR、各milestone固定图与逐样本FEM、最终HTML和exact FEM binary arrays。
'''
    (HERE / 'REPORT.md').write_text(text, encoding='utf-8')
    (HERE / 'START_HERE.md').write_text('先读REPORT.md与training_curves.png。final_evaluation/index.html可直接打开，展示全部生成图和两组训练集机械检索。run/milestone_metrics.json与milestone_evaluations/提供阶段对照，fixed_seed_*.png是不筛选的同seed全24目标。run/config.json为实际最终配置；budget_boundary_200000/保存原预算边界记录，不是最终停止。run/training.jsonl保留全部原始loss/LR；checkpoint_inventory.json列出未入包权重路径和SHA256。可运行python audit_verify.py --base . --complete独立核对核心数值（需numpy）。AUDIT_FILES.json为包内内容hash清单。\n', encoding='utf-8')
    files = []
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py', '.md', '.png']: files.append((p, p.name))
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix not in ['.pt', '.tmp', '.zip', '.pyc'] and '__pycache__' not in p.parts and p.name != 'audit_package.json':
            files.append((p, p.relative_to(HERE).as_posix()))
    for p in out.rglob('*'):
        if p.is_file() and 'arrays' not in p.relative_to(out).parts and p.name != 'initial_noise.npy':
            files.append((p, 'final_evaluation/' + p.relative_to(out).as_posix()))
    assert len({name for _, name in files}) == len(files)
    save(HERE / 'AUDIT_FILES.json', dict(files=[dict(path=name, bytes=p.stat().st_size, sha256=sha(p)) for p, name in files]))
    files.append((HERE / 'AUDIT_FILES.json', 'AUDIT_FILES.json'))
    package = HERE / 'cfm_v1_ai_review.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, name in files: z.write(p, name)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('AUDIT_FILES.json'))
        import hashlib
        for r in manifest['files']: assert hashlib.sha256(z.read(r['path'])).hexdigest() == r['sha256']
    save(RUN / 'audit_package.json', dict(path=str(package), bytes=package.stat().st_size, sha256=sha(package), crc_check='passed', all_member_hashes='passed'))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
