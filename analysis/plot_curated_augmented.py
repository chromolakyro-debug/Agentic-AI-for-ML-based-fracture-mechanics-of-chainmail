"""Curated eight-style D4 coverage: 8 scatters + 1 heatmap in each coordinate system.

Reuse the original metadata checks, coarse Ex/Ey bins and necessary envelope.
Only analysis files are written; D4 labels use the previously FEM-validated mapping.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_hex

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/'cat_binary_dataset_curated_v1'
OUT=HERE/'curated_augmented_distribution'
sys.path.insert(0,str(ROOT/'augmentation_validation'))
sys.path.insert(0,str(ROOT/'cfm_v0'))
from augmentation import map_ex_ey
from common import to_sa,sha_file
from compare_augmentation import load_metadata,COARSE_EDGES,wholly_excluded_cells,describe

SEED=2026111602
JITTER=.015  # bounded display-only offset in normalized Ex/Ey, 0.167% of axis span
LOG10=np.log(10.)
AMAX=float(np.log(121/40))
S_EDGES=np.linspace(0,LOG10,7)
A_EDGES=np.linspace(-AMAX,AMAX,7)
# Heatmaps only: wider sparse tails, finer central cells. Scatter axes stay fixed.
HEAT_E_EDGES=np.array([1.,4.,5.,6.,7.,10.])
HEAT_S_EDGES=np.array([0.,.75,1.1,1.45,1.8,LOG10])
HEAT_A_EDGES=np.array([-AMAX,-.3,-.1,.1,.3,AMAX])


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def a_bound(s):
    # x*y+10 <= 11*min(x,y), x=exp(s+a/2), y=exp(s-a/2).
    return 2*np.log(11*np.exp(s)/(np.exp(2*s)+10))


def in_envelope(xy):
    x,y=xy.T
    return (x>=1)&(x<=10)&(y>=1)&(y<=10)&(y>=10/(11-x)-1e-12)&(y<=11-10/x+1e-12)


def style_axes(space):
    fig=plt.figure(figsize=(8.5,8.8),facecolor='white')
    ax=fig.add_axes([.13,.18,.78,.66])
    if space=='ex_ey':
        x=np.linspace(1,10,1000)
        ax.set_facecolor('#f0f1f2');ax.fill_between(x,10/(11-x),11-10/x,color='white',zorder=0)
        ax.plot(x,10/(11-x),color='#9fb0be',lw=.85);ax.plot(x,11-10/x,color='#9fb0be',lw=.85)
        ax.plot([1,10],[1,10],ls='--',lw=.7,color='#a0aab4',zorder=1)
        ax.set(xlim=(1,10),ylim=(1,10),xlabel=r'$E_x/E_{\mathrm{soft}}$',ylabel=r'$E_y/E_{\mathrm{soft}}$')
        ax.set_xticks(COARSE_EDGES);ax.set_yticks(COARSE_EDGES)
        ax.set_aspect('equal',adjustable='box')
    else:
        s=np.linspace(0,LOG10,1000);b=a_bound(s)
        ax.set_facecolor('#f0f1f2');ax.fill_between(s,-b,b,color='white',zorder=0)
        ax.plot(s,-b,color='#9fb0be',lw=.85);ax.plot(s,b,color='#9fb0be',lw=.85)
        ax.axhline(0,color='#a0aab4',ls='--',lw=.7,zorder=1)
        ax.set(xlim=(0,LOG10),ylim=(-AMAX,AMAX),xlabel=r'$s=\frac{1}{2}[\log(E_x/E_{\mathrm{soft}})+\log(E_y/E_{\mathrm{soft}})]$',ylabel=r'$a=\log(E_x/E_y)$')
        ax.set_xticks(S_EDGES);ax.set_xticklabels([f'{v:.2f}' for v in S_EDGES])
        ax.set_yticks(A_EDGES);ax.set_yticklabels([f'{v:.2f}' for v in A_EDGES])
        ax.set_box_aspect(1)
    ax.grid(color='#e8edf1',lw=.55);ax.set_axisbelow(True)
    return fig,ax


def scatter(space,style,values,source_count,color):
    fig,ax=style_axes(space)
    ax.scatter(values[:,0],values[:,1],s=8,c=color,alpha=.34,linewidths=0,rasterized=True,zorder=3)
    fig.text(.105,.947,f'{style["group_id"]}  {style["group_name_zh"]}',fontsize=20,color=color)
    fig.text(.105,.912,style['group_name_en']+' | Joint distribution',fontsize=14)
    fig.text(.105,.873,f'{source_count:,} source images  ×  8 D4 transforms  =  {len(values):,} plotted records',fontsize=10.7,color='#536373')
    fig.text(.105,.090,'All curated samples pooled; identity is one of the eight transforms.',fontsize=9,color='#536373')
    fig.text(.105,.064,'Small fixed-seed jitter for display only; 8 records do not mean 8 distinct labels.',fontsize=9,color='#536373')
    fig.text(.105,.038,'Gray: outside necessary material bounds. White is not proof of cat-shape feasibility.',fontsize=8.7,color='#617080')
    path=OUT/space/f'{style["group_id"]}_{style["group_label"]}.png'
    fig.savefig(path,dpi=200,facecolor='white');plt.close(fig)
    return path


def sa_excluded(s_edges=S_EDGES,a_edges=A_EDGES):
    excluded=np.zeros((len(a_edges)-1,len(s_edges)-1),bool)
    for iy,(ya,yb) in enumerate(zip(a_edges[:-1],a_edges[1:])):
        min_abs=0. if ya<=0<=yb else min(abs(ya),abs(yb))
        for ix,(sa,sb) in enumerate(zip(s_edges[:-1],s_edges[1:])):
            peak=np.clip(LOG10/2,sa,sb)
            excluded[iy,ix]=min_abs>a_bound(peak)+1e-12
    return excluded


def heatmap(space,raw):
    edges=(HEAT_E_EDGES,HEAT_E_EDGES) if space=='ex_ey' else (HEAT_S_EDGES,HEAT_A_EDGES)
    counts=np.histogram2d(raw[:,0],raw[:,1],bins=edges)[0].T.astype(int)
    excluded=wholly_excluded_cells(HEAT_E_EDGES) if space=='ex_ey' else sa_excluded(*edges)
    assert counts.sum()==len(raw) and counts[excluded].sum()==0
    fig,ax=style_axes(space);ax.clear();ax.set_position([.105,.19,.70,.64])
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#d6dce1')
    heat=ax.pcolormesh(*edges,np.ma.masked_where(excluded,counts),cmap=cmap,vmin=0,vmax=int(counts.max()),edgecolors='white',lw=.9)
    # Show the physical boundary inside partially intersecting coarse cells.
    boundary_x=np.linspace(edges[0][0],edges[0][-1],1000)
    if space=='ex_ey':boundaries=[10/(11-boundary_x),11-10/boundary_x]
    else:boundaries=[a_bound(boundary_x),-a_bound(boundary_x)]
    for yy in boundaries:ax.plot(boundary_x,yy,color='#899ba8',lw=.9,ls='--',alpha=.75)
    for iy in range(len(edges[1])-1):
        for ix in range(len(edges[0])-1):
            xx=edges[0][ix:ix+2].mean();yy=edges[1][iy:iy+2].mean();n=counts[iy,ix]
            ax.text(xx,yy,'×' if excluded[iy,ix] else f'{n:,}',ha='center',va='center',fontsize=12 if excluded[iy,ix] else 10,
                color='#89949e' if excluded[iy,ix] or n==0 else ('white' if n>.58*counts.max() else '#214765'))
    ax.set(xlim=(edges[0][0],edges[0][-1]),ylim=(edges[1][0],edges[1][-1]))
    ax.set_box_aspect(1);ax.set_xticks(edges[0]);ax.set_yticks(edges[1])
    if space=='ex_ey':ax.set(xlabel=r'$E_x/E_{\mathrm{soft}}$',ylabel=r'$E_y/E_{\mathrm{soft}}$')
    else:
        ax.set(xlabel=r'$s=\frac{1}{2}[\log(E_x/E_{\mathrm{soft}})+\log(E_y/E_{\mathrm{soft}})]$',ylabel=r'$a=\log(E_x/E_y)$')
        ax.set_xticklabels([f'{v:.2f}' for v in edges[0]]);ax.set_yticklabels([f'{v:.2f}' for v in edges[1]])
    cax=fig.add_axes([.845,.243,.026,.53]);bar=fig.colorbar(heat,cax=cax);bar.set_label('D4 records per bin',fontsize=10);bar.outline.set_visible(False)
    fig.text(.105,.947,'八类合并 / All eight styles',fontsize=20)
    fig.text(.105,.912,'Joint counts | 5 × 5 unequal-width bins',fontsize=14)
    fig.text(.105,.873,f'7,857 source images  ×  8 transforms  =  {len(raw):,} records',fontsize=10.7,color='#536373')
    fig.text(.105,.090,'Wider sparse tails; counts use exact labels before jitter. Counts are not densities.',fontsize=9,color='#536373')
    fig.text(.105,.064,'Dashed curves: necessary bounds. A populated cell is not supported everywhere.',fontsize=9,color='#536373')
    fig.text(.105,.038,'D4 repetitions are included; no new data. Previous 6 × 6 grids are archived.',fontsize=8.7,color='#617080')
    path=OUT/space/'09_all_styles_counts.png';fig.savefig(path,dpi=200,facecolor='white');plt.close(fig)
    save(OUT/space/'counts.json',dict(x_edges=edges[0].tolist(),y_edges=edges[1].tolist(),counts_y_rows_x_columns=counts.tolist(),
        wholly_excluded=excluded.tolist(),total=int(counts.sum()),occupied_bins=int((counts>0).sum()),jitter_used=False,
        grid='5x5 unequal-width; wider sparse tails',minimum_count=int(counts.min()),
        cell_area=np.outer(np.diff(edges[1]),np.diff(edges[0])).tolist(),density_normalized=False))
    return path,counts


def main():
    start=time.perf_counter()
    for space in ('ex_ey','s_a'):(OUT/space).mkdir(parents=True,exist_ok=True)
    path,rows,values=load_metadata(DATA)
    classes=read(DATA/'classes.json')['classes'];frozen=np.load(DATA/'data.npz',allow_pickle=False)
    checked=read(ROOT/'augmentation_validation/results/summary.json');assert checked['complete_pass'] and checked['mapping_pass']==160
    assert len(rows)==7857 and len({r['new_id'] for r in rows})==7857
    physical=np.array([[float(r['Ex_app']),float(r['Ey_app'])] for r in rows]);soft=np.array([float(r['E_soft']) for r in rows])
    assert np.array_equal(physical,frozen['moduli'])
    assert np.array_equal(np.array([r['new_id'] for r in rows]),frozen['sample_ids'])
    xy=physical/soft[:,None];assert np.allclose(xy,values[:,:2],rtol=1e-14,atol=1e-14)
    assert np.allclose(to_sa(xy[:,0],xy[:,1]),frozen['sa'],rtol=1e-14,atol=1e-14)
    protected=[DATA/p for p in ('metadata.csv','data.npz','dataset.npy','splits.json','index.csv')]
    protected+=[ROOT/'analysis/reference_expansion/reference_expansion_comparison.png',ROOT/'analysis/compare_reference_expansion.py',ROOT/'analysis/compare_augmentation.py']
    before={p.relative_to(ROOT).as_posix():sha_file(p) for p in protected}
    source_index=np.repeat(np.arange(len(rows)),8)
    k=np.tile(np.repeat(np.arange(4),2),len(rows));flip=np.tile([False,True],len(rows)*4)
    augmented=np.empty((len(source_index),2),np.float64)
    for turns in range(4):
        mask=k==turns
        xx,yy=map_ex_ey(xy[source_index[mask],0],xy[source_index[mask],1],turns)
        augmented[mask]=np.column_stack([xx,yy])
    sa=to_sa(augmented[:,0],augmented[:,1])
    assert in_envelope(augmented).all()
    assert np.allclose(np.abs(sa[:,1]),np.abs(frozen['sa'][source_index,1]),rtol=1e-13,atol=1e-14)
    assert np.allclose(sa[:,0],frozen['sa'][source_index,0],rtol=1e-13,atol=1e-14)
    assert len(augmented)==62856 and np.array_equal(np.bincount(source_index),np.full(7857,8))
    rng=np.random.default_rng(SEED)
    # Every mapped pair has >=4 exact coincident records. Center per-pair jitter
    # so its display centroid remains exactly at the actual material label.
    _,inverse,duplicate_counts=np.unique(augmented,axis=0,return_inverse=True,return_counts=True)
    offsets=rng.uniform(-1,1,augmented.shape)
    for axis in range(2):offsets[:,axis]-=np.bincount(inverse,weights=offsets[:,axis])[inverse]/duplicate_counts[inverse]
    offsets*=JITTER/max(1,float(np.max(np.abs(offsets))))
    # Shrink whole coincident groups near necessary bounds; do not drop records.
    scale=np.ones(len(duplicate_counts));shrunk=set()
    for _ in range(40):
        shown=augmented+offsets*scale[inverse,None];bad=~in_envelope(shown)
        if not bad.any():break
        groups=np.unique(inverse[bad]);scale[groups]*=.5;shrunk.update(groups.tolist())
    assert in_envelope(shown).all() and np.max(np.abs(shown-augmented))<=JITTER+1e-14
    shown_sa=to_sa(shown[:,0],shown[:,1])  # same displayed points, only coordinates change
    group_ids=np.array([r['group_id'] for r in rows]);group_aug=group_ids[source_index]
    np.savez_compressed(OUT/'plot_data.npz',source_index=source_index,quarter_turns=k,horizontal_flip=flip,
        group_id=group_aug,Ex_over_Esoft=augmented[:,0],Ey_over_Esoft=augmented[:,1],s=sa[:,0],a=sa[:,1],
        displayed_Ex=shown[:,0],displayed_Ey=shown[:,1],displayed_s=shown_sa[:,0],displayed_a=shown_sa[:,1])
    csvsave(OUT/'source_samples.csv',[dict(source_index=i,new_id=r['new_id'],original_id=r['original_id'],group_id=r['group_id'],
        group_name_en=r['group_name_en'],new_relative_path=r['new_relative_path'],split=r['curation.split'],sha256_pixels=r['sha256_pixels'],
        Ex_app=r['Ex_app'],Ey_app=r['Ey_app'],E_soft=r['E_soft']) for i,r in enumerate(rows)])
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
        'text.color':'#253342','axes.labelcolor':'#253342','axes.edgecolor':'#617080','axes.unicode_minus':False,'mathtext.fontset':'dejavusans'})
    palette=plt.get_cmap('tab10');colors={c['group_id']:to_hex(palette(i)) for c,i in zip(classes,[0,1,2,3,4,5,6,9])}
    figures=[];style_stats=[]
    for c in classes:
        select=group_aug==c['group_id'];original=group_ids==c['group_id'];assert original.sum()==c['count']
        assert select.sum()==8*c['count']
        for space,points in [('ex_ey',shown),('s_a',shown_sa)]:
            figures.append(scatter(space,c,points[select],int(original.sum()),colors[c['group_id']]))
        style_stats.append(dict(group_id=c['group_id'],style=c['group_name_en'],count=int(original.sum()),augmented_records=int(select.sum()),
            unique_exact_property_pairs=int(len(np.unique(augmented[select],axis=0))),color=colors[c['group_id']],
            s_min=float(sa[select,0].min()),s_median=float(np.median(sa[select,0])),s_max=float(sa[select,0].max()),
            absolute_a_p90=float(np.quantile(np.abs(sa[select,1]),.9)),absolute_a_max=float(np.max(np.abs(sa[select,1])))))
    matrices={}
    for space,raw in [('ex_ey',augmented),('s_a',sa)]:
        p,counts=heatmap(space,raw);figures.append(p);matrices[space]=counts
    assert np.array_equal(matrices['ex_ey'],matrices['ex_ey'].T)
    assert np.array_equal(matrices['s_a'],matrices['s_a'][::-1])
    assert len(figures)==18 and sum(len(list((OUT/space).glob('*.png'))) for space in ('ex_ey','s_a'))==18
    for p,digest in before.items():assert sha_file(ROOT/p)==digest
    csvsave(OUT/'style_statistics.csv',style_stats)
    summary=dict(source_dataset=str(DATA),source_images=7857,transform_records=62856,number_of_figures=18,
        unique_exact_property_pairs=int(len(duplicate_counts)),exact_property_pair_multiplicity=dict(zip(*[a.tolist() for a in np.unique(duplicate_counts,return_counts=True)])),
        source_identity='Explicit new_id joined to frozen data.npz sample_ids and moduli; all eight styles pooled across current splits',
        augmentation='Enumerate all8 D4 transforms, identity included once; CCW rotation then horizontal flip. Only odd quarter-turns swap Ex/Ey.',
        augmentation_label_provenance='Analytical symmetry mapping validated by augmentation_validation; NOT 62856 new independent FEM solves or unique images.',
        sa_definition='s=0.5*(ln(Ex/Esoft)+ln(Ey/Esoft)), a=ln(Ex/Ey); raw, not standardized',
        jitter=dict(seed=SEED,max_normalized_modulus_offset=JITTER,observed_max_offset=float(np.max(np.abs(shown-augmented))),
            exact_coincident_groups_centered=True,groups_shrunk_at_envelope=len(shrunk),
            sa_display='Convert the same jittered Ex/Ey points to s/a; not a second independent perturbation',histograms_use_jitter=False),
        bins=heatmap_bins(),
        sa_envelope='abs(a)<=2*ln(11*exp(s)/(exp(2*s)+10)); 0<=s<=ln(10); necessary bounds only',
        symmetry=dict(Ex_Ey_histogram_transpose=True,sa_histogram_a_reflection=True),
        counts_y_rows_x_columns={s:c.tolist() for s,c in matrices.items()},styles=style_stats,
        distributions=dict(s=describe(sa[:,0]),absolute_a=describe(np.abs(sa[:,1])),
            middle_a_bins_share=float(np.mean(np.abs(sa[:,1])<A_EDGES[4]))),
        source_files_sha256=before,source_files_unchanged=True,plot_script_sha256=sha_file(__file__),
        reused_sources=['analysis/compare_augmentation.py','augmentation_validation/augmentation.py','cfm_v0/common.py'],
        fem_recomputed=False,training_started=False,elapsed_seconds=time.perf_counter()-start,
        command='.\\cfm_v0\\.venv\\Scripts\\python.exe analysis\\plot_curated_augmented.py')
    save(OUT/'statistics.json',summary)
    report(summary)
    gallery(classes)
    save(OUT/'figure_manifest.json',[dict(path=p.relative_to(OUT).as_posix(),sha256=sha_file(p)) for p in sorted(figures)])
    print(json.dumps(dict(source_images=7857,augmented_records=62856,unique_property_pairs=len(duplicate_counts),
        figures=18,output=str(OUT),elapsed_seconds=summary['elapsed_seconds'],styles=style_stats),ensure_ascii=False,indent=2))


def report(d):
    strongest=max(d['styles'],key=lambda r:r['absolute_a_p90'])
    lines=['# 八类数据集与D4 augmentation的物性分布','',
        '复查原图analysis/reference_expansion/reference_expansion_comparison.png及analysis/compare_reference_expansion.py后，仅保留其A联合散点与B二维计数两种图。每套为八类各一张单色散点、全部八类合并的一张计数热图；Ex/Ey与s/a共18张当前PNG，见index.html。热图现为5×5非等宽网格，原6×6等宽版本保存在grid_v1_equal_width/。没有original/extended的分层、颜色区分或前后箭头。','',
        '数据来自整理后的7,857张，按new_id逐条核对data.npz的sample_ids及moduli，跨现有train/validation/test合并作描述性统计。每张枚举先CCW旋转0°/90°/180°/270°、后左右翻转否/是的全部8种D4变换；恒等计一次，不额外重复原图，共62,856条等权augmentation记录。','',
        f'Ex/Ey使用上一轮真实FEM验证通过的映射；本次没有重新计算FEM或生成增强图副本。这62,856条记录只有{d["unique_exact_property_pairs"]:,}组精确不同的Ex/Ey坐标，每张源图通常只贡献(Ex,Ey)与(Ey,Ex)两组，各重复4次。因此数量增加不等于同等数量的独立图像、独立FEM求解或新物理结构。', '',
        '散点只加入显示用jitter：seed=2026111602，归一化Ex/Ey每轴最大偏移0.015，约为横纵轴全跨度的0.167%。每组精确重合点的扰动均值为零；临近必要物理外界时整组缩小扰动，不丢弃记录。同一组已扰动Ex/Ey通过公式转为s/a，因此两套散点使用相同点及扰动，只换坐标。原始标签、统计量及热图不加jitter。','',
        's=0.5[ln(Ex/Esoft)+ln(Ey/Esoft)]，a=ln(Ex/Ey)，使用自然对数、未标准化。D4保持s和|a|不变，奇数个90°使a变号；不会补出新的总体刚度水平或更强的各向异性。Ex/Ey分布关于对角线、s/a分布关于a=0对称，此对称来自完整D4枚举，不代表原始样本天然对称。','',
        '| 类别 | 源图片数 | D4记录数 | s中位数 | abs(a) P90 |','|---|---:|---:|---:|---:|']
    for r in d['styles']:lines.append(f'| {r["group_id"]} {r["style"]} | {r["count"]:,} | {r["augmented_records"]:,} | {r["s_median"]:.3f} | {r["absolute_a_p90"]:.3f} |')
    lines+=['',f'各风格的分布差异保留；按|a| P90衡量，{strongest["style"]}的各向异性尾部最宽（{strongest["absolute_a_p90"]:.3f}）。固定指标|a|<{A_EDGES[4]:.3f}包含{d["distributions"]["middle_a_bins_share"]:.2%}的记录；此指标沿用旧网格阈值，不是新版中心格范围。完整D4枚举没有消除较强各向异性区域的样本稀疏。这只是描述性分布统计，不是生成质量或可控性结论。','',
        '图2改为5×5非等宽网格，加宽外围稀疏区。Ex/Ey共同边界为[1,4,5,6,7,10]；s边界为[0,0.75,1.10,1.45,1.80,ln10]；a边界为[−ln(121/40),−0.30,−0.10,0.10,0.30,ln(121/40)]。坐标范围保持不变。当前两图各25格均有记录，最少分别20和72条D4记录。计数总和均为62,856。网格根据本数据的分布作描述性调整，不是预注册的支持阈值或训练分箱。','',
        '非等宽格子的面积不同，颜色与数字展示每格记录数，不代表单位面积密度。格子加宽只是汇总了邻近区域，不能解释为新增样本或稀疏区域处处受支持；必要物理边界以虚线保留。旧图、旧counts.json与报告保留于grid_v1_equal_width/，具体前后统计见GRID_UPDATE.md与grid_update.json。','',
        '保留原图必要物理外界：10/(11−Ex/Esoft)≤Ey/Esoft≤11−10/(Ex/Esoft)。换算到s/a为|a|≤2ln[11exp(s)/(exp(2s)+10)]。灰色/×只表示整个区域或格子被该必要界排除，白色区域不等于猫形可达域，未观测的空格也不等于物理不可达。','',
        '验证：18张图完整输出；每张源图恰有8条记录、4条不交换与4条交换；标签与源data.npz一致；两热图计数守恒与预期对称性通过；源metadata/NPY/NPZ/splits及旧图和脚本hash未变。本次无训练、无新的FEM求解。','',
        '文件：ex_ey/与s_a/各9张PNG及counts.json；plot_data.npz含真实/显示坐标、源索引、旋转和flip；source_samples.csv保存源ID/路径/split；style_statistics.csv与statistics.json保存分析；figure_manifest.json保存图像hash。','',
        '复现：从项目根目录执行`.\\cfm_v0\\.venv\\Scripts\\python.exe analysis\\plot_curated_augmented.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def gallery(classes):
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>八类物性分布 / D4 augmentation</title>',
        '<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#eef2f5;color:#253342;margin:24px}h1{font-size:24px}a{color:#2465a0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}figure{margin:0;background:white;border-radius:8px;padding:8px}img{width:100%;height:auto}figcaption{padding:6px}nav{position:sticky;top:0;background:#eef2f5;padding:10px;z-index:2}section{scroll-margin-top:65px}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}}</style>',
        '<h1>八类物性分布：7,857张 × 8种D4变换 = 62,856条记录</h1><p>原始与新增批次合并。散点jitter仅用于显示；热图统计未扰动的映射标签。</p>',
        '<nav><a href="#ex_ey">Ex–Ey：9张</a>　 <a href="#s_a">s–a：9张</a>　 <a href="REPORT.md">报告</a></nav>']
    for space,label in [('ex_ey','Ex–Ey'),('s_a','s–a')]:
        parts.append(f'<section id="{space}"><h2>{label}</h2><div class="grid">')
        for c in classes:
            path=f'{space}/{c["group_id"]}_{c["group_label"]}.png';caption=c['group_id']+' '+c['group_name_zh']
            parts.append(f'<figure><a href="{path}"><img src="{path}" alt="{label} {caption}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>')
        path=f'{space}/09_all_styles_counts.png'
        parts.append(f'<figure><a href="{path}"><img src="{path}" alt="八类合并计数" loading="lazy"></a><figcaption>09 八类合并计数</figcaption></figure></div></section>')
    parts.append('</html>');(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')


def heatmap_bins():
    return dict(ex_ey_edges=HEAT_E_EDGES.tolist(),s_edges=HEAT_S_EDGES.tolist(),a_edges=HEAT_A_EDGES.tolist(),
        note='5x5 unequal-width descriptive bins; sparse tails widened. Counts are not densities. Scatter axes remain unchanged.')


def update_heatmaps():
    start=time.perf_counter();d=np.load(OUT/'plot_data.npz',allow_pickle=False)
    protected=read(OUT/'grid_change_protected_files.json')
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
        'text.color':'#253342','axes.labelcolor':'#253342','axes.edgecolor':'#617080','axes.unicode_minus':False,'mathtext.fontset':'dejavusans'})
    results={};comparison={}
    for space,raw in [('ex_ey',np.c_[d['Ex_over_Esoft'],d['Ey_over_Esoft']]),('s_a',np.c_[d['s'],d['a']])]:
        _,counts=heatmap(space,raw);results[space]=counts.tolist()
        old=read(OUT/'grid_v1_equal_width'/space/'counts.json');oldc=np.array(old['counts_y_rows_x_columns'])
        comparison[space]=dict(old_grid=[6,6],new_grid=list(counts.shape),old_nonzero_bins=int((oldc>0).sum()),
            new_nonzero_bins=int((counts>0).sum()),old_minimum_positive_count=int(oldc[oldc>0].min()),
            new_minimum_count=int(counts.min()),old_total=int(oldc.sum()),new_total=int(counts.sum()))
        assert counts.sum()==62856 and (counts>0).all()
    assert np.array_equal(results['ex_ey'],np.array(results['ex_ey']).T)
    assert np.array_equal(results['s_a'],np.array(results['s_a'])[::-1])
    stats=read(OUT/'statistics.json');stats['bins']=heatmap_bins();stats['counts_y_rows_x_columns']=results
    stats['heatmap_revision']=dict(version=2,elapsed_seconds=time.perf_counter()-start,plot_script_sha256=sha_file(__file__),
        command='.\\cfm_v0\\.venv\\Scripts\\python.exe analysis\\plot_curated_augmented.py --heatmaps-only')
    save(OUT/'statistics.json',stats);report(stats)
    manifest=read(OUT/'figure_manifest.json')
    for r in manifest:
        if r['path'].endswith('09_all_styles_counts.png'):r['sha256']=sha_file(OUT/r['path'])
    save(OUT/'figure_manifest.json',manifest)
    for p,digest in protected.items():assert sha_file(OUT/p)==digest,('Unrelated file changed',p)
    for p,digest in stats['source_files_sha256'].items():assert sha_file(ROOT/p)==digest
    save(OUT/'grid_update.json',dict(comparison=comparison,bins=heatmap_bins(),
        preserved_file_count=len(protected),source_data_unchanged=True,jitter_used=False,
        note='Bins chosen descriptively after inspecting occupancy; no samples created or moved. D4 repeats are not independent specimens.'))
    text=['# 图2网格调整','',
        '两套图2由6×6等宽改为5×5非等宽，外围稀疏区采用更宽的格子。坐标范围、全部62,856条记录和真实标签不变；不使用jitter计数。16张风格散点、60张例图及原数据均未改变。','',
        '| 坐标系 | 旧版有样本格数 | 新版有样本格数 | 旧版最少非零计数 | 新版最少计数 |','|---|---:|---:|---:|---:|']
    for name,r in comparison.items():text.append(f'| {name} | {r["old_nonzero_bins"]}/36 | {r["new_nonzero_bins"]}/25 | {r["old_minimum_positive_count"]} | {r["new_minimum_count"]} |')
    text+=['','Ex/Ey边界：[1,4,5,6,7,10]。s边界：[0,0.75,1.10,1.45,1.80,ln10]。a边界：[−ln(121/40),−0.30,−0.10,0.10,0.30,ln(121/40)]。','',
        '这是根据现有分布调整的描述性网格。不同格子面积不同，颜色为总计数而非密度；格内部分区域仍可能没有样本。20或72条D4记录也不等于20或72个独立原图。所有格有样本不能证明整个矩形可达或适合训练，图中保留必要物理边界虚线。','',
        '旧版图、counts、报告和脚本均存于grid_v1_equal_width/。当前图片路径不变，总览页直接显示新版。']
    (OUT/'GRID_UPDATE.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--heatmaps-only',action='store_true');args=parser.parse_args()
    if args.heatmaps_only:update_heatmaps()
    else:main()
