"""Sample source cats from low/high modulus clusters, retaining exact PNG bytes."""
from pathlib import Path
from collections import Counter
import csv
import json
import hashlib
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'cat_binary_dataset_curated_v1'
OUT=ROOT/'analysis/curated_augmented_distribution/low_high_examples'
# Append new classes after the original four: retain their exact RNG draws.
GROUPS=('02','03','05','07','06','08')
SEED=2026111603


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def region(g):return 'low' if g<=2.5 else ('high' if g>=4 else 'middle')
def describe(rr):
    if not rr:return dict(n=0)
    return dict(n=len(rr),g_min=min(r['g'] for r in rr),g_max=max(r['g'] for r in rr),
        g_median=float(np.median([r['g'] for r in rr])),
        black_fraction_median=float(np.median([float(r['black_fraction']) for r in rr])),
        black_fraction_quantiles=np.quantile([float(r['black_fraction']) for r in rr],[0,.1,.5,.9,1]).tolist(),
        background_value_counts=dict(Counter(r['background_value'] for r in rr)),
        fine_style_counts=dict(Counter(r['requested_style'] for r in rr)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((OUT/'samples.json').read_text(encoding='utf-8')) if (OUT/'samples.json').exists() else []
    old_members=[r for r in prior if r['group_id'] in GROUPS[:4]]
    old_files={r['copied_relative_path']:sha(OUT/r['copied_relative_path']) for r in old_members}
    for p in OUT.glob('*_comparison.png'):
        if p.name[:2] in GROUPS[:4]:old_files[p.name]=sha(p)
    if len(prior)==40:
        archive=OUT/'previous_four_class_snapshot';archive.mkdir(exist_ok=True)
        for name in ('REPORT.md','samples.csv','samples.json','config.json','region_statistics.json','original_metadata.jsonl','verification.json','index.html'):
            if not (archive/name).exists():shutil.copy2(OUT/name,archive/name)
    with (DATA/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    labels=np.load(DATA/'data.npz',allow_pickle=False)
    protected=[DATA/p for p in ('metadata.csv','metadata.jsonl','data.npz','splits.json')]
    before={p.name:sha(p) for p in protected}
    classes={c['group_id']:c for c in json.loads((DATA/'classes.json').read_text(encoding='utf-8'))['classes']}
    rows=[]
    for i,r in enumerate(original):
        if r['group_id'] not in GROUPS:continue
        assert r['new_id']==labels['sample_ids'][i]
        assert np.array_equal([float(r['Ex_app']),float(r['Ey_app'])],labels['moduli'][i])
        g=float(np.sqrt(float(r['Ex_app'])*float(r['Ey_app']))/float(r['E_soft']))
        rows.append(dict(r,dataset_index=i,g=g,s=float(np.log(g)),region=region(g)))
    rng=np.random.default_rng(SEED);selected=[];summary={}
    for gid in GROUPS:
        group=[r for r in rows if r['group_id']==gid];c=classes[gid]
        summary[gid]=dict(name=c['group_name_zh'],count=len(group),
            regions={k:describe([r for r in group if r['region']==k]) for k in ('low','middle','high')},
            by_background={bg:describe([r for r in group if r['background_value']==bg]) for bg in ('0','1')},
            pearson_g_black_fraction=float(np.corrcoef([r['g'] for r in group],[float(r['black_fraction']) for r in group])[0,1]))
        summary[gid]['middle_share']=summary[gid]['regions']['middle']['n']/len(group)
        summary[gid]['g_histogram_half_unit_counts']=np.histogram([r['g'] for r in group],bins=np.arange(1,10.5,.5))[0].tolist()
        summary[gid]['g_histogram_edges']=np.arange(1,10.5,.5).tolist()
        for band in ('low','high'):
            candidates=[r for r in group if r['region']==band]
            assert len(candidates)>=5
            picks=rng.choice(len(candidates),size=5,replace=False)
            folder=OUT/(gid+'_'+c['group_label'])/band;folder.mkdir(parents=True,exist_ok=True)
            for order,pick in enumerate(picks,1):
                r=candidates[int(pick)];src=DATA/r['new_relative_path'];dst=folder/(r['new_id']+'.png')
                shutil.copy2(src,dst);assert sha(src)==sha(dst)
                with Image.open(dst) as im:
                    assert im.size==(224,224)
                    arr=np.asarray(im,dtype=np.uint8)
                    assert np.isin(arr,[0,1]).all()
                    assert hashlib.sha256(arr.tobytes()).hexdigest()==r['sha256_pixels']
                    assert abs(float((arr==0).mean())-float(r['black_fraction']))<1e-12
                selected.append(dict(group_id=gid,group_name_zh=c['group_name_zh'],group_label=c['group_label'],region=band,selection_order=order,
                    dataset_index=r['dataset_index'],new_id=r['new_id'],original_id=r['original_id'],
                    source_relative_path=r['new_relative_path'],copied_relative_path=dst.relative_to(OUT).as_posix(),
                    split=r['curation.split'],Ex_app=float(r['Ex_app']),Ey_app=float(r['Ey_app']),g=r['g'],s=r['s'],
                    black_fraction=float(r['black_fraction']),background_value=int(r['background_value']),
                    requested_style=r['requested_style'],png_sha256=sha(dst),sha256_pixels=r['sha256_pixels']))
    assert len(selected)==len({r['new_id'] for r in selected})==10*len(GROUPS)
    if old_members:assert selected[:40]==old_members,'Original40 selections must not change'
    csvsave(OUT/'samples.csv',selected);save(OUT/'samples.json',selected)
    by_id={r['new_id']:r for r in original}
    with (OUT/'original_metadata.jsonl').open('w',encoding='utf-8') as f:
        for r in selected:f.write(json.dumps(dict(new_id=r['new_id'],original_metadata=by_id[r['new_id']]),ensure_ascii=False)+'\n')
    save(OUT/'config.json',dict(seed=SEED,selection='Uniform without replacement, 5 source images per class/region; not selected by visual appearance',
        groups=list(GROUPS),low='g<=2.5',high='g>=4.0',middle='2.5<g<4.0; retained in statistics, not in examples',
        region_basis='Common descriptive cluster windows, chosen from previously plotted distribution and coarse Ex/Ey grid; not an optimized clustering classifier',
        g='sqrt(Ex_app*Ey_app)/Esoft = exp(s); unchanged by D4 augmentation',
        source_metadata_sha256=before['metadata.csv'],augmentation_used=False,image_copy='Original PNG bytes; no rotation, inversion, resizing or re-encoding'))
    save(OUT/'region_statistics.json',summary)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    for gid in GROUPS:
        c=classes[gid];canvas=Image.new('RGB',(1230,885),'#f3f5f7');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),f'{gid} {c["group_name_zh"]} | 低模量与高模量样本',font=font(28),fill='#173345')
        draw.text((20,55),'g = sqrt(Ex × Ey) / Esoft；每个区域固定随机抽5张原图；0=黑色硬材料，1=白色软材料',font=font(18),fill='#50616b')
        for j,band in enumerate(('low','high')):
            y=96+j*367
            n=summary[gid]['regions'][band]['n'];label='低模量区：g ≤ 2.5' if band=='low' else '高模量区：g ≥ 4.0'
            draw.text((20,y),f'{label}  |  候选 {n} 张，展示5张',font=font(22),fill='#173345')
            for k,r in enumerate([r for r in selected if r['group_id']==gid and r['region']==band]):
                x=20+k*242;yy=y+40
                with Image.open(OUT/r['copied_relative_path']) as im:canvas.paste(im.convert('RGB'),(x,yy))
                draw.rectangle((x-1,yy-1,x+224,yy+224),outline='#ccd2d7')
                draw.text((x,yy+229),r['new_id'],font=font(17),fill='#173345')
                draw.text((x,yy+255),f'Ex={r["Ex_app"]:.3f}  Ey={r["Ey_app"]:.3f}',font=font(16),fill='#173345')
                draw.text((x,yy+280),f'g={r["g"]:.3f}  黑色硬相={r["black_fraction"]:.1%}',font=font(16),fill='#50616b')
        draw.text((20,844),f'seed={SEED}；同一源图只展示一次；未以augmentation副本充数；源文件及split不变。',font=font(17),fill='#50616b')
        canvas.save(OUT/(gid+'_'+c['group_label']+'_comparison.png'))
    lines=['# 六类低/高模量样本：追加像素实心与剪影','',
        '每个类别、每个区域无放回随机抽5张，共60张不同源图，seed=2026111603。原四类40张样本及四张对照图逐文件保留不变，新追加像素实心、剪影各10张。六张对照图每张上排低模量、下排高模量；对应原PNG字节副本在各类别的low/与high/目录。samples.csv给出ID、路径、原有split、Ex/Ey、g及黑色比例，original_metadata.jsonl完整保存所选样本metadata。原四类清单与说明另存previous_four_class_snapshot/。','',
        '选区统一使用g=√(Ex·Ey)/Esoft=exp(s)：低区g≤2.5，高区g≥4，中间2.5<g<4不展示、但保留在统计中。窗口用于挑出已观察到的两个集中区，不声称是唯一聚类边界。只从未变换的源图抽样；D4不改变g，augmentation不会制造该双峰。','',
        '| 类别 | 低区数量 | 低区硬相中位数 | 高区数量 | 高区硬相中位数 | 中间区数量（占本类） |','|---|---:|---:|---:|---:|---:|']
    for gid in GROUPS:
        r=summary[gid];lo=r['regions']['low'];hi=r['regions']['high']
        lines.append(f'| {r["name"]} | {lo["n"]} | {lo["black_fraction_median"]:.1%} | {hi["n"]} | {hi["black_fraction_median"]:.1%} | {r["regions"]["middle"]["n"]}（{r["middle_share"]:.1%}） |')
    lines+=['','原四类的解释仍成立：写实笔绘、几何三角、像素线描、卡通的全部低区候选background_value=1（白色软相背景），全部高区候选为0（黑色硬相背景）。这是四类完整候选池的统计，不只是展示图；其g与黑色硬相占比的Pearson相关系数为0.970–0.988。','',
        '像素实心、剪影并非完全没有背景相位效应，而是两部分分布有重叠，且中间模量区样本较多。像素实心中间区313/766=40.9%，剪影239/1082=22.1%，原四类仅0%–6.2%。这解释了散点中两团之间的空档为何不明显；没有进行正式的单峰/双峰统计检验，不能把视觉上更连续说成严格单峰。','',
        '像素实心的低区硬相中位数41.9%，高区70.2%，两者差距比原四类小。白色背景的硬相比例范围11.5%–64.9%，黑色背景37.5%–94.6%，存在明显重叠；两种背景的g范围分别为1.224–3.772和2.185–8.637。实心主体占据较多像素，使白底猫也能包含较多硬材料、黑底猫也可能被较大的软材料主体占据，符合上述相位比例统计。低区已有15张黑背景样本，中间区也同时有208张黑背景、105张白背景。','',
        '剪影的模量与材料占比覆盖更宽：白背景g为1.079–3.906，黑背景为2.308–9.333，低区存在3张黑背景。它合并11个原细类，造型、尺度与实心占比差异可进一步拓宽分布；但不能只归因于合并标签，因为solid_silhouette、bold_ink、paper_cut、rounded_cartoon、soft_folk、angular_graphic各自也都有中间区样本。region_statistics.json保留按背景、按原细类计数及相位比例分位数。','',
        'reference_style_candidates.py:140在每个style下同时安排background=0与1；:296先用background填满材料图，再把foreground赋为1−background。mechanics.py固定0→Ehard=10、1→Esoft=1。因此白底黑猫通常是在软材料基体中放入较少硬材料，黑底白猫通常是在大量硬材料中放入软材料猫形，材料体积分数及承载拓扑随之变化。这足以解释两个主要集中区的来源；不能将黑白互换当作保持力学标签的显示反色。','',
        '以上直接确认了中间区占比、背景相位和硬相比例的区别；“实心造型、尺度和连通性如何分别造成差异”仍是机制解释，尚未通过控制变量FEM实验分离。背景色不是模量的充分决定因素，也不应把低区和高区样本认作逐对互补图。D4保持s和硬材料比例，不会消除或制造此处的总体刚度分布差异。','',
        '验证：60张源图ID唯一；所有复制PNG的SHA256及pixel hash与源记录一致；每区域恰5张；原40张选择和四张对照图hash不变；Ex/Ey逐项匹配data.npz；源metadata、data.npz和splits hash未变。没有新FEM求解、图像生成或模型训练。','',
        '复现：从项目根目录执行`.\\cfm_v0\\.venv\\Scripts\\python.exe analysis\\extract_low_high_examples.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    html=['<!doctype html><meta charset="utf-8"><title>低/高模量样本</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1250px;margin:24px auto;color:#173345;background:#eef2f5}img{width:100%;height:auto}section{margin:20px 0}a{color:#2465a0}</style><h1>六类低/高模量样本：每区域5张</h1><p>上排：g≤2.5；下排：g≥4；原40张不变，新增像素实心与剪影20张，共60张不同原图。<a href="#06">像素实心</a> · <a href="#08">剪影</a> · <a href="REPORT.md">分析说明</a> · <a href="samples.csv">样本清单</a></p>']
    for gid in GROUPS:
        c=classes[gid];file=gid+'_'+c['group_label']+'_comparison.png'
        html.append(f'<section id="{gid}"><h2>{c["group_name_zh"]}</h2><a href="{file}"><img src="{file}" alt="{c["group_name_zh"]}低高模量各5张"></a></section>')
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    for name,digest in before.items():assert sha(DATA/name)==digest
    for name,digest in old_files.items():assert sha(OUT/name)==digest,('Old figure/sample changed',name)
    save(OUT/'verification.json',dict(status='passed',source_samples=len(selected),copied_pngs=len(selected),comparison_sheets=len(GROUPS),
        original_four_class_files_unchanged=old_files,
        all_file_sha256_match=True,all_pixel_sha256_match=True,all_moduli_match=True,source_hashes_unchanged=before,
        seed=SEED,code_sha256=sha(__file__)))
    print(json.dumps(dict(output=str(OUT),samples=len(selected),comparison_sheets=len(GROUPS),
        original_files_preserved=len(old_files),middle_share={gid:summary[gid]['middle_share'] for gid in GROUPS}),indent=2))


if __name__=='__main__':main()
