"""Measured continuation results, numeric checks and compact AI review package."""
import json,zipfile,math
import numpy as np
from common import ROOT,HERE,RUN,read,save,sha
from curves import curves

def main():
    curves();cfg=read(RUN/'config.json');summary=read(RUN/'training_summary.json')
    final=read(RUN/'final_evaluation.json');out=ROOT/final['output'];rows=read(out/'results.json')
    ok=[r for r in rows if r['status']=='ok'];assert len(rows)==240
    vals={r['step']:r for r in map(json.loads,(RUN/'training.jsonl').read_text(encoding='utf-8').splitlines()) if r['kind']=='validation'}
    best=min(vals,key=lambda s:vals[s]['validation_velocity_mse']);assert best==summary['best_step']
    records=[json.loads(line) for line in (RUN/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    train={r['step']:r for r in records if r['kind']=='train'}
    assert set(train)==set(range(1,summary['step']+1))
    for step,r in train.items():
        c,pos=divmod(step-1,20000);peak=max(.0001,.0002*.5**c)
        lr=.00003+(peak-.00003)*.5*(1+math.cos(math.pi*pos/19999))
        if c==0:lr*=min(step/200,1)
        assert abs(lr-r['lr'])<1e-16
    parent=read(RUN/'parent_plateau_history.json');hist=read(RUN/'plateau_history.json');assert hist[:len(parent)]==parent
    prior=min(r['cycle_best'] for r in parent);count=0;armed=None
    for r in hist[len(parent):]:
        value=min(v['validation_velocity_mse'] for s,v in vals.items() if r['step']-20000<s<=r['step'])
        improvement=(prior-value)/max(abs(prior),1e-12)
        assert improvement==r['relative_improvement'];count=count+1 if improvement<.001 else 0
        if improvement>=.001:armed=None
        stop=False
        if count>=4:
            if armed is None:armed=r['cycle']
            else:stop=r['cycle']>=armed+1
        assert (count,armed,stop)==(r['stop_patience_count'],r['stop_armed_cycle'],r['stop'])
        prior=min(prior,value)
    if summary['reason']=='convergence_criterion':assert hist[-1]['stop']
    elif summary['reason']=='update_budget':assert summary['step']==cfg['limits']['max_updates']
    else:assert summary['training_seconds']>=cfg['limits']['max_training_seconds']
    binary=[];arrays={}
    for r in rows:
        if r['raw_array'] not in arrays:arrays[r['raw_array']]=np.load(out/r['raw_array'])['binary']
        b=arrays[r['raw_array']][r['array_index']];binary.append(b)
        import hashlib
        assert hashlib.sha256(b.tobytes()).hexdigest()==r['input_pixel_sha256']==r['binary_pixel_sha256']
        if r['status']=='ok':assert np.isclose(np.hypot(r['Ex_FEM']-r['Ex_target'],r['Ey_FEM']-r['Ey_target']),r['d_mech'],rtol=1e-13)
    np.savez_compressed(RUN/'final_fem_inputs.npz',binary=np.stack(binary),sample_keys=[r['sample_key'] for r in rows])
    metrics=dict(valid=len(ok),failed=len(rows)-len(ok),mean_d=float(np.mean([r['d_mech'] for r in ok])),
        hits_sa=sum(r['hit_s_a'] for r in ok),hits_exey=sum(r['hit_Ex_Ey'] for r in ok))
    save(RUN/'final_metrics.json',metrics);save(RUN/'verification.json',dict(status='passed',best_validation=True,LR_all_steps=True,new_plateau_recomputed=True,FEM_binary_identity=True))
    inventory=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in list((RUN/'checkpoints').glob('*.pt'))+list((RUN/'evaluation_checkpoints').glob('*.pt'))]
    save(RUN/'checkpoint_inventory.json',inventory)
    history=[]
    for e in read(RUN/'evaluations.json'):
        rr=read(ROOT/e['output']/'results.json');valid=[r for r in rr if r['status']=='ok']
        history.append(dict(**e,valid=len(valid),mean_d=float(np.mean([r['d_mech'] for r in valid])),hits_sa=sum(r['hit_s_a'] for r in valid)))
    save(RUN/'evaluation_history.json',history)
    extra_hours=(summary['training_seconds']-cfg['continuation']['prior_training_seconds'])/3600
    text=f'''# CFM v1 continuation 1

从v1 280k完整last恢复，训练至{summary['step']:,}，新增{summary['step']-280000:,} updates，本轮active训练/validation/checkpoint耗时{extra_hours:.3f}小时；停止原因`{summary['reason']}`。原v1数据、checkpoint和日志保留。新输出集中cfm_v1_cont1。

模型、AdamW moments、EMA、scheduler及RNG恢复检查逐项一致；恢复后的固定validation与原280k差约1.7e-8，两次复算完全一致。11.53M U-Net、batch32、bfloat16、增强、train-only归一化及oversampling沿用v1，固定划分6285/786/786不变。固定validation noise/time，每5k计算EMA velocity MSE，定义为`mean[(v_theta-(x-epsilon))²]`，不是FEM力学误差。

停止门槛由每20k周期改善<0.5%连续3次，改为<0.1%连续4次，之后仍需额外完整restart周期；≥0.1%改善清空严格计数。只清空旧停止计数，不删除旧历史或重置训练状态。续训从300k周期开始判定；已激活evaluation，每20k进行完整240样本FEM评估并留存完整checkpoint。LR按原绝对step继续，峰1e-4、谷3e-5，每20k restart一次，不清空optimizer moments。本轮安全上限额外200k updates或12小时active时间。

最终按best-validation选择step {summary['best_step']:,}，MSE={summary['best_val_mse']:.9f}；原280k为0.096081788。最终240张真实二值FEM有效{metrics['valid']}、失败{metrics['failed']}，mean(d_mech)={metrics['mean_d']:.6f}，s/a联合命中{metrics['hits_sa']}/240、Ex/Ey ±5%联合命中{metrics['hits_exey']}/240。`d_mech=sqrt((Ex_FEM-Ex_target)^2+(Ey_FEM-Ey_target)^2)`；s/a容限为log(1.05)、log(1.10)。原v1 280k对应mean(d)=0.320352、s/a=120/240。该24×10 panel为复用历史panel，不是全新holdout，FEM不用于调参或选择checkpoint。

逐阶段结果见run/evaluation_history.json，所有图像与两组train-only机械retrieval在`{final['output']}/index.html`。曲线保留原280k历史并继续追加。视觉质量不能由MSE或FEM替代，待固定样本实际查看后补充观察；没有宣称memorization或多样性已通过量化检验。

恢复：`cfm_v0/.venv/Scripts/python.exe cfm_v1_cont1/workflow.py`（未完成时从last继续；完成后仅复核最终评估）。配置、命令、hash与完整状态均在run中；审计ZIP不含大权重和训练图片。
'''
    (HERE/'REPORT.md').write_text(text,encoding='utf-8')
    (HERE/'START_HERE.md').write_text('先读REPORT.md和training_curves.png；final_evaluation/index.html可直接打开。run/config.json为续训配置，parent_config保存父配置。run/training.jsonl前280k为原v1历史，之后为本次新增训练。weights不入包，checkpoint_inventory.json列路径和hash。\n',encoding='utf-8')
    files=[]
    for p in HERE.glob('*'):
        if p.is_file() and p.suffix in ['.py','.md','.png']:files.append((p,p.name))
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix not in ['.pt','.tmp','.pyc','.zip'] and '__pycache__' not in p.parts and p.name!='audit_package.json':files.append((p,p.relative_to(HERE).as_posix()))
    for p in out.rglob('*'):
        if p.is_file() and 'arrays' not in p.relative_to(out).parts and p.name!='initial_noise.npy':files.append((p,'final_evaluation/'+p.relative_to(out).as_posix()))
    save(HERE/'AUDIT_FILES.json',dict(files=[dict(path=n,sha256=sha(p),bytes=p.stat().st_size) for p,n in files]));files.append((HERE/'AUDIT_FILES.json','AUDIT_FILES.json'))
    package=HERE/'cfm_v1_cont1_ai_review.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p,n in files:z.write(p,n)
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    save(RUN/'audit_package.json',dict(path=str(package),bytes=package.stat().st_size,sha256=sha(package)))
    print(summary,metrics)

if __name__=='__main__':main()
