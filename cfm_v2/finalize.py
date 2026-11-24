"""Single training figure, concise measured report and a compact external-AI audit ZIP."""
import argparse,csv,json,zipfile,datetime
from pathlib import Path
import numpy as np
from common import HERE,ROOT,RUN,read,save,sha

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def curves():
    rows=[json.loads(x) for x in (RUN/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['kind']=='train'];val=[r for r in rows if r['kind']=='validation']
    # On an abnormal resume, the trailing pre-checkpoint log can contain replayed updates.
    # Preserve all raw records; plotted/CSV view takes the last record for each step.
    train=list({r['step']:r for r in train}.values());train.sort(key=lambda r:r['step'])
    val=list({r['step']:r for r in val}.values());val.sort(key=lambda r:r['step'])
    write_csv(RUN/'training_history.csv',train);write_csv(RUN/'validation_history.csv',val)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    x=np.array([r['step'] for r in train]);y=np.array([r['loss'] for r in train])
    ax[0].plot([r['step'] for r in val],[r['validation_velocity_mse'] for r in val],'.-',label='fixed validation EMA velocity MSE');ax[0].set_ylabel('Validation MSE')
    ax[1].plot(x,y,color='#6699c0',lw=.25,alpha=.25,label='raw per-update train MSE')
    if len(y)>=100:ax[1].plot(x[99:],np.convolve(y,np.ones(100)/100,mode='valid'),lw=.7,label='100-update mean (display only)')
    ax[1].set_ylabel('Train MSE');ax[1].set_yscale('log')
    ax[2].plot(x,[r['lr'] for r in train],color='#b15b25');ax[2].set_ylabel('Learning rate');ax[2].set_xlabel('Optimizer updates')
    for a in ax:
        a.grid(alpha=.2)
        for step in range(20000,int(x.max())+1,20000):a.axvline(step,ls=':',lw=.6,color='.4')
    ax[0].legend();ax[1].legend();fig.suptitle('CFM v2 (no augmentation) — velocity MSE and cosine warm restarts (not mechanical error)');fig.tight_layout();fig.savefig(HERE/'training_curves.png',dpi=170);plt.close(fig)
    return train,val


def comparison_curves(val):
    with (RUN/'reference_v1/validation_history.csv').open(encoding='utf-8-sig', newline='') as f:
        reference = {int(r['step']): float(r['validation_velocity_mse']) for r in csv.DictReader(f)}
    paired = [dict(step=r['step'], v1_validation_mse=reference[r['step']],
                   v2_validation_mse=r['validation_velocity_mse'],
                   relative_v2_minus_v1=r['validation_velocity_mse']/reference[r['step']]-1)
              for r in val if r['step'] in reference]
    write_csv(RUN/'paired_validation.csv', paired)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(list(reference), list(reference.values()), '.-', label='v1: D4 augmentation')
    axes[0].plot([r['step'] for r in val], [r['validation_velocity_mse'] for r in val], '.-', label='v2: no augmentation')
    axes[0].set_ylabel('Fixed EMA validation velocity MSE'); axes[0].legend()
    axes[1].plot([r['step'] for r in paired], [100*r['relative_v2_minus_v1'] for r in paired], '.-')
    axes[1].axhline(0, color='.5', lw=.7); axes[1].set_ylabel('v2 / v1 - 1 (%) at same step'); axes[1].set_xlabel('Optimizer updates')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('CFM augmentation ablation: same validation protocol, one paired training seed')
    fig.tight_layout(); fig.savefig(HERE/'ablation_curves.png', dpi=170); plt.close(fig)
    return paired


def metrics(rows):
    ok = [r for r in rows if r['status'] == 'ok']
    return dict(images=len(rows), valid=len(ok), failed=len(rows)-len(ok),
                mean_d_mech=float(np.mean([r['d_mech'] for r in ok])) if ok else None,
                joint_s_a_hits=sum(r['hit_s_a'] for r in ok), joint_Ex_Ey_hits=sum(r['hit_Ex_Ey'] for r in ok))


def audit_retention():
    from retention import POLICY
    entries = [json.loads(line) for line in (RUN/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    vals = sorted({r['step']: r for r in entries if r['kind']=='validation'}.values(), key=lambda r:r['step'])
    trigger = next((b['step'] for a,b in zip(vals, vals[1:])
                    if abs(b['validation_velocity_mse']-a['validation_velocity_mse'])/
                    max(abs(a['validation_velocity_mse']), POLICY['epsilon']) < POLICY['threshold']), None)
    end = read(RUN/'training_summary.json')['step']
    expected = [] if trigger is None else list(range(((trigger+19999)//20000)*20000, end+1, 20000))
    retained = read(RUN/'retained_checkpoints.json') if (RUN/'retained_checkpoints.json').exists() else []
    assert [r['step'] for r in retained] == expected
    import torch
    required = {'model','ema','optimizer','scheduler','plateau','rng','step','training_seconds','retention','config'}
    for r in retained:
        path = ROOT/r['path']; assert sha(path)==r['sha256']
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert required <= set(state) and state['step']==r['step'] and state['retention']['trigger_step']==trigger
        del state
    return dict(status='passed', trigger_step=trigger, retained_steps=expected, full_state_and_hashes_verified=True)


def finalize():
    import shutil
    from audit_verify import verify
    from visual_review import render
    train, val = curves(); paired = comparison_curves(val)
    cfg = read(RUN/'config.json'); end = read(RUN/'training_summary.json')
    final = read(RUN/'final_evaluation.json'); out = ROOT/final['output']
    rows = read(out/'results.json'); score = metrics(rows); save(RUN/'final_metrics.json', score)
    reference = RUN/'reference_v1'; previous_end = read(reference/'training_summary.json')
    previous_final = read(reference/'final_evaluation.json'); previous_out = ROOT/previous_final['output']
    previous_rows = read(previous_out/'results.json'); previous_score = metrics(previous_rows)
    assert read(out/'targets.json') == read(previous_out/'targets.json')
    assert read(out/'seeds.json') == read(previous_out/'seeds.json')
    previous_lookup = {(r['target_id'], r['seed']): r for r in previous_rows}
    paired_mechanics = []
    for r in rows:
        p = previous_lookup[(r['target_id'], r['seed'])]
        paired_mechanics.append(dict(target_id=r['target_id'], seed=r['seed'], v1_step=previous_end['best_step'],
            v2_step=end['best_step'], v1_status=p['status'], v2_status=r['status'],
            v1_d_mech=p.get('d_mech'), v2_d_mech=r.get('d_mech'), v1_hit_s_a=p.get('hit_s_a'), v2_hit_s_a=r.get('hit_s_a')))
    write_csv(RUN/'paired_final_mechanics.csv', paired_mechanics)
    for name in ['results.json','summary.json','config.json','targets.json','seeds.json']:
        shutil.copyfile(previous_out/name, reference/f'final_{name}')
    milestones = []
    for event in read(RUN/'evaluations.json'):
        if any(r['step']==event['checkpoint_step'] for r in milestones): continue
        path = ROOT/event['output']; row = dict(step=event['checkpoint_step'], validation_mse=event['validation_mse'], **metrics(read(path/'results.json')))
        milestones.append(row)
        dest = RUN/'milestone_evaluations'/str(row['step']); dest.mkdir(parents=True, exist_ok=True)
        for name in ['results.json','summary.json','config.json','targets.json','seeds.json','manifest.json']:
            if (path/name).exists(): shutil.copyfile(path/name,dest/name)
    save(RUN/'milestone_metrics.json', milestones)
    # Package exact binary FEM inputs, without including the full training images or weights.
    binary, arrays = [], {}
    for r in rows:
        if r['raw_array'] not in arrays: arrays[r['raw_array']] = np.load(out/r['raw_array'], allow_pickle=False)['binary']
        binary.append(arrays[r['raw_array']][r['array_index']])
    np.savez_compressed(RUN/'final_fem_inputs.npz', binary=np.stack(binary), sample_keys=np.array([r['sample_key'] for r in rows]))
    inventory = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                 for folder in ['checkpoints','evaluation_checkpoints'] for p in (RUN/folder).glob('*.pt')]
    save(RUN/'checkpoint_inventory.json', inventory)
    audit = verify(HERE, complete=True); audit['retention'] = audit_retention()
    save(RUN/'independent_complete_audit.json', audit)
    render(end['best_step'])
    history = read(RUN/'plateau_history.json')
    stage_a = next((r['step'] for r in history if r['started_evaluation']), None)
    table = '\n'.join(f"| {r['step']:,} | {r['validation_mse']:.7f} | {r['mean_d_mech']:.4f} | {r['joint_s_a_hits']}/{r['images']} |" for r in milestones)
    retention = audit['retention']
    report = f'''# CFM v2 无 data augmentation 消融

实际训练至 {end['step']:,} updates，active训练/validation/checkpoint耗时 {end['training_seconds']/3600:.3f} 小时。停止原因 `{end['reason']}`；最佳 EMA 在 {end['best_step']:,}，validation velocity MSE={end['best_val_mse']:.9f}。实际 LR restart {end['restart_count']} 次。最终评估仅按 validation 选择。

与 CFM v1 相比，训练干预仅为关闭 D4 图像旋转/镜像及对应 Ex/Ey 交换。完全复用相同 7,857 张数据、6285/786/786 划分、训练集 normalization、逐样本 oversampling weights、seed 20261117、11.53M Conditional U-Net、batch32、AdamW、EMA、bfloat16、loss、LR 和 plateau 判据。从随机初始化开始。配置差异机器校验仅 version、augmentation、sampling.order；原数据和 v1 未改动。

保留未使用的 rotation/flip RNG 抽样，检查证实模型实际输入为原图、物理条件未交换，初始化、样本抽样及noise/time随机流与v1对齐。跨restart恢复检查模型权重完全一致，固定validation重复差异为0。cuDNN benchmark继续沿用v1，不宣称历史GPU运算bitwise完全复现。这是单个training seed的配对实验，不是多seed统计结论。

每5k对全部786张固定validation计算EMA velocity MSE，不做增强；loss是像素均值 `(v_theta-(x-epsilon))²`，不是FEM误差。test未用于训练、选择或停止。LR每20k一个cosine周期，首峰2e-4、200-step warmup，后续峰1e-4、谷3e-5，不重置optimizer moments。

80k后比较完整周期最低validation与此前历史最低值；连续2周期改善<1%启动evaluation，连续3周期<0.5%进入待停止，再完成至少一个完整restart周期仍无≥0.5%改善才停止；有效改善重置严格patience。采用v1最终300k/18小时active预算。阶段A触发step={stage_a}。训练停止步数可随数据增强改变，不将各自停止后的比较称作固定训练量比较。

checkpoint独立留存遵循用户修正后的规则：相邻validation的绝对相对变化首次<1%时激活，仅在20k整数倍steps保存。触发step={retention['trigger_step']}，实际留存steps={retention['retained_steps']}；完整状态和SHA256核验通过。last仍每1k更新，best_val_mse按最优validation保存。

| best EMA checkpoint | validation MSE | mean(d_mech) | s/a联合命中 |
| --- | --- | --- | --- |
{table}

最终固定24 targets × 10 seeds全部使用64-step midpoint及阈值0二值化，真实FEM有效{score['valid']}/{score['images']}、失败{score['failed']}，s/a联合命中{score['joint_s_a_hits']}/{score['images']}，Ex/Ey各±5%联合命中{score['joint_Ex_Ey_hits']}/{score['images']}。s/a阈值分别为log(1.05)、log(1.10)。FEM失败计入命中率分母，mean(d_mech)仅统计有效求解。

v1最佳step={previous_end['best_step']}、validation MSE={previous_end['best_val_mse']:.9f}、mean(d_mech)={previous_score['mean_d_mech']:.6f}、s/a命中{previous_score['joint_s_a_hits']}/{previous_score['images']}；v2最佳step={end['best_step']}、mean(d_mech)={score['mean_d_mech']:.6f}、s/a命中{score['joint_s_a_hits']}/{score['images']}。paired_validation.csv提供相同训练步数的validation比较；paired_final_mechanics.csv按target/seed配对各自最佳模型。完整曲线见ablation_curves.png。固定panel已有历史使用，不是新holdout，机械结果未反向选择checkpoint。

完整生成图及train-only retrieval见 `{final['output']}/index.html`。固定seed全24目标、不筛选的图见run/fixed_seed_{end['best_step']:06d}.png。数值审计不等于猫形视觉质量确认；不从机械误差直接推断视觉质量、创新性或memorization。retrieval设置沿用v1，包含其允许的轴交换规则。

恢复入口 `python cfm_v2/workflow.py`，实际命令在run/commands.jsonl。审计ZIP包含配置、代码、完整原始loss/LR、逐样本FEM、最终HTML、binary FEM输入、v1参考结果与hash清单，不包含权重或全训练图。无自动Git保存。
'''
    (HERE/'REPORT.md').write_text(report, encoding='utf-8')
    (HERE/'START_HERE.md').write_text('先读REPORT.md、training_curves.png和ablation_curves.png。final_evaluation/index.html展示生成图和retrieval。run/ablation_protocol.json与reference_v1/记录对照；paired_validation.csv比较相同步数；paired_final_mechanics.csv比较各自最佳模型。运行python audit_verify.py --base . --complete可核验核心数值。模型权重不在ZIP，路径与SHA256见checkpoint_inventory.json。\n', encoding='utf-8')
    files = []
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py','.md','.png']: files.append((p,p.name))
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix not in ['.pt','.tmp','.pyc','.zip'] and '__pycache__' not in p.parts and p.name!='audit_package.json':
            files.append((p,p.relative_to(HERE).as_posix()))
    for p in out.rglob('*'):
        if p.is_file() and 'arrays' not in p.relative_to(out).parts and p.name!='initial_noise.npy':
            files.append((p,'final_evaluation/'+p.relative_to(out).as_posix()))
    save(HERE/'AUDIT_FILES.json', dict(files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in files]))
    files.append((HERE/'AUDIT_FILES.json','AUDIT_FILES.json'))
    package = HERE/'cfm_v2_ai_review.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in files:z.write(p,name)
    import hashlib
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for item in read(HERE/'AUDIT_FILES.json')['files']:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    save(RUN/'audit_package.json', dict(path=str(package),bytes=package.stat().st_size,sha256=sha(package),crc_check='passed',all_member_hashes='passed'))
    print(json.dumps(dict(summary=end,final_metrics=score),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--curves-only',action='store_true');args=parser.parse_args()
    if args.curves_only:
        _,val=curves();comparison_curves(val)
    else:finalize()
