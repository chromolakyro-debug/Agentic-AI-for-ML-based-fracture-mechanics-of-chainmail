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
    ax[0].legend();ax[1].legend();fig.suptitle('CFM v1 — velocity MSE and cosine warm restarts (not mechanical error)');fig.tight_layout();fig.savefig(HERE/'training_curves.png',dpi=170);plt.close(fig)
    return train,val

def finalize():
    train,val=curves();cfg=read(RUN/'config.json');summary=read(RUN/'training_summary.json');final=read(RUN/'final_evaluation.json');out=ROOT/final['output']
    rows=read(out/'results.json');valid=[r for r in rows if r['status']=='ok'];evals=read(RUN/'evaluations.json');periodic=[r for r in evals if r['reason']=='periodic_after_stage_A']
    scores=dict(images=len(rows),valid=len(valid),failed=len(rows)-len(valid),mean_d_mech=float(np.mean([r['d_mech'] for r in valid])) if valid else None,
        joint_s_a_hits=sum(r['hit_s_a'] for r in valid),joint_Ex_Ey_hits=sum(r['hit_Ex_Ey'] for r in valid))
    save(RUN/'final_metrics.json',scores)
    inventory=[]
    for p in list((RUN/'checkpoints').glob('*.pt'))+list((RUN/'evaluation_checkpoints').glob('*.pt')):inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    save(RUN/'checkpoint_inventory.json',inventory)
    periodic_text=f'阶段A首次在{read(RUN/"plateau_history.json")[next(i for i,r in enumerate(read(RUN/"plateau_history.json")) if r["started_evaluation"])]["step"]} updates启动，完成{len(periodic)}次周期评估。' if periodic else '训练期间未触发阶段A；最终仍完整评估best_val_mse。'
    comparison=[]
    for record in evals:
        rr=read(ROOT/record['output']/'results.json');ok=[r for r in rr if r['status']=='ok']
        comparison.append(dict(**record,valid=len(ok),images=len(rr),mean_d_mech=float(np.mean([r['d_mech'] for r in ok])) if ok else None,joint_s_a_hits=sum(r['hit_s_a'] for r in ok),joint_Ex_Ey_hits=sum(r['hit_Ex_Ey'] for r in ok)))
    save(RUN/'evaluation_history.json',comparison);write_csv(RUN/'evaluation_history.csv',comparison)
    old=read(ROOT/'evaluation/comparison/overall.json')[1]
    text=f'''# CFM v1训练与评估

实际训练至{summary['step']:,} optimizer updates，累计{summary['samples_seen']:,}个训练样本呈现，等效{summary['equivalent_epochs']:.2f} epochs（有放回加权采样，不代表每张图都被遍历）。active训练/validation/checkpoint耗时{summary['training_seconds']/3600:.3f}小时；停止原因为`{summary['reason']}`。周期restart实际发生{summary['restart_count']}次。

沿用11,529,633参数Conditional U-Net、s/a FiLM、Gaussian linear flow matching、velocity MSE、AdamW(weight_decay=.01)、EMA=.999、gradient clip=1、batch32和bfloat16。新模型从随机初始化开始，没有加载旧模型。检查实测吞吐{read(RUN/'checks.json')['images_per_second']:.1f} images/s，GPU为{read(RUN/'checks.json')['gpu']}，测试峰值分配显存{read(RUN/'checks.json')['peak_allocated_GiB']:.2f} GiB；实际训练显存及吞吐全记录在training.jsonl。

新数据版本curated八类v1，7,857张，固定train/validation/test=6285/786/786。完整核对分类、old/new IDs、PNG/NPY、metadata及力学标签；删除的143张未参与采样。每类基础概率按样本数量，沿用原始训练样本s/a的8×12分箱逆平方根频数与均匀混合，截断后均值归一；权重逐个关联新train IDs，未复制旧权重。用户批准仅在新训练集重新拟合z-score mean/std，原数据未改动。原始归一化值及新的实际配置均留在data_audit/config中。

每次取样先按上述权重抽原图，再uniform CCW旋转0/90/180/270度，独立50%水平翻转；先旋转后翻转。奇数90度在物理Ex/Ey上交换，重新计算`s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]`、`a=log(Ex/Ey)`后才标准化。没有插值、缩放、反色或永久8倍副本，没有增强后的二次重加权。validation不增强，使用786张固定顺序和固定noise/time，每5000 updates评估EMA。loss是`mean[(v_theta-(x-epsilon))^2]`，其中`x=2*binary-1`；无额外loss项，不是FEM机械误差。test没有参与训练选择。

LR按update进行cosine restart：每周期20,000 steps，T_mult=1；初始峰值2e-4，200-step warmup；之后峰值1e-4并保持，min_lr=3e-5。不重置模型或optimizer moments，只保留一个scheduler。完整scheduler、EMA、optimizer、RNG、plateau及进度随last恢复；跨restart边界恢复测试的下一step损失、索引、增强及模型权重完全一致，固定validation重复差异为0。

两阶段判据预先冻结：80k后按完整20k周期计算`(best_before-cycle_best)/max(abs(best_before),1e-12)`；连续2周期<1%启动evaluation，连续3周期<0.5%只armed，之后还必须完整经历至少一个restart周期，仍无≥0.5%改善才可停止。有效改善重置严格patience。兜底为200k或12小时active训练时间。{periodic_text} 全部判定窗口与计数见plateau_history.json，不把预算停止称作收敛。

最佳EMA来自step {summary['best_step']:,}，validation velocity MSE={summary['best_val_mse']:.9f}；最终评估使用该best_val_mse，而非默认last。最终240张固定targets/seeds中FEM有效{scores['valid']}/240，失败{scores['failed']}；mean(d_mech)={scores['mean_d_mech']:.6f}，Ex/Ey各±5%联合命中{scores['joint_Ex_Ey_hits']}/240，`|Δs|≤log(1.05)`且`|Δa|≤log(1.10)`联合命中{scores['joint_s_a_hits']}/240。距离为`d_mech=sqrt((Ex_FEM-Ex_target)^2+(Ey_FEM-Ey_target)^2)`。第三行retrieval query是FEM-selected best-of-10，不等同随机生成整体表现。

作为历史参考，旧125k在同24 targets/10 seeds上的mean(d)为{old['mean_d_mech']:.6f}、s/a联合命中{old['hits_s_a']}/240。新旧训练数据及划分不同，不能当作单一augmentation因素的消融。猫形可辨识性、多样性及memorization通过既有HTML人工判断；本轮没有自动视觉打分，也不把机械近邻作为视觉近邻。完整每格结果及train-only retrieval在最终页面内，validation MSE与FEM指标分开保存。

最终评估页面：`{final['output']}/index.html`。实际命令见run/commands.jsonl；恢复入口为`python cfm_v1/workflow.py`。last保留完整训练状态，best_val_mse与各次evaluation的不可变快照均列于checkpoint_inventory.json，含保存路径和SHA256。训练曲线见training_curves.png，原始逐step记录保留，平滑仅用于展示。审计包包含完整历史及最终二值图/FEM结果，不含完整训练图片或大型模型权重。
'''
    (HERE/'REPORT.md').write_text(text,encoding='utf-8')
    (HERE/'START_HERE.md').write_text('先读REPORT.md和training_curves.png；run/config.json记录冻结设置，run/training.jsonl包含所有原始loss/LR，run/plateau_history.json解释停止。final_evaluation/index.html是全部24×10结果及两组训练集机械检索，人工判读外观。模型权重不入包，run/checkpoint_inventory.json给出路径与hash。\n',encoding='utf-8')
    # Export exact FEM binary input without duplicating continuous raw outputs in the small archive.
    binary=[];arrays={}
    for r in rows:
        if r['raw_array'] not in arrays:arrays[r['raw_array']]=np.load(out/r['raw_array'],allow_pickle=False)['binary']
        binary.append(arrays[r['raw_array']][r['array_index']])
    np.savez_compressed(RUN/'final_fem_inputs.npz',binary=np.stack(binary),sample_keys=np.array([r['sample_key'] for r in rows]))
    files=[]
    for p in HERE.glob('*'):
        if p.is_file() and p.suffix in ['.py','.md','.png']:files.append((p,p.relative_to(HERE).as_posix()))
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix not in ['.pt','.tmp'] and '__pycache__' not in p.parts:files.append((p,p.relative_to(HERE).as_posix()))
    for p in out.rglob('*'):
        if p.is_file() and 'arrays' not in p.relative_to(out).parts and p.name!='initial_noise.npy':files.append((p,'final_evaluation/'+p.relative_to(out).as_posix()))
    manifest=dict(files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in files]);save(HERE/'AUDIT_FILES.json',manifest);files.append((HERE/'AUDIT_FILES.json','AUDIT_FILES.json'))
    path=HERE/'cfm_v1_ai_review.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in files:z.write(p,name)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    save(RUN/'audit_package.json',dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),crc_check='passed'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--curves-only',action='store_true');args=parser.parse_args()
    curves() if args.curves_only else finalize()
