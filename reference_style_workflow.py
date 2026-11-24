"""Append reference-guided cat styles with actual FEM and pixel-bound AI review.

No training or split is performed. The original 1250 specimens are immutable.
Scratch contains candidates/review sheets; release contains the required outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import candidate_generation as base
import augmentation_workflow as previous
import reference_style_candidates as engine
import reference_style_selection as selection
from evaluate_mechanics import label_items, mechanics_fingerprint, initialize_worker, solve_item
from release_workflow import contact_sheet

BASE_COUNT=1250
BASE_IDENTITY='d453530f264aa8475f150ced7ae11f745291f4c7e9862922285c5c6d69a2eb1a'
BASE_ARRAY_FILE_SHA='ab806034466a9a75afd8775f91a475ab841b010663dd58e46d854f205eeedf47'
REFERENCE_STYLE_REVIEW={}


def digest(pixels):return hashlib.sha256(np.asarray(pixels,dtype=np.uint8).tobytes()).hexdigest()
def key(sample):return f"reference:{sample.spec.seed}"
def read_json(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(obj,indent=2),encoding='utf-8');temporary.replace(path)
def read_rows(folder):
    with (folder/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def identity(rows):
    payload=[[r['filename'],r['sha256_pixels'],str(r['seed']),float(r['Ex_app']),float(r['Ey_app'])] for r in rows]
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def load_base(folder):
    samples,labels,rows=selection.load_base(folder)
    if len(rows)!=BASE_COUNT or identity(rows)!=BASE_IDENTITY:raise ValueError('Original1250 identity mismatch')
    pixels=np.load(folder/'dataset.npy',allow_pickle=False)
    if pixels.shape!=(BASE_COUNT,224,224) or pixels.dtype!=np.uint8:raise ValueError('Wrong base array')
    if any(digest(p)!=r['sha256_pixels'] for p,r in zip(pixels,rows)):raise ValueError('Base pixel mismatch')
    return samples,labels,rows,pixels


def review(args):
    data=read_json(args.work_dir/'reference_visual_review.json',REFERENCE_STYLE_REVIEW)
    return set(data.get('passed_hashes',[])),data.get('excluded_hashes',{})


def cache_paths(args,stage):
    return sorted(engine.cache_dir(stage,args.work_dir,args.seed,args.batch_size).glob('batch_*.npz'))


def read_sources(args,stage):
    state=read_json(args.work_dir/f'{stage}_generation_state.json',{})
    if not state.get('complete'):raise ValueError(f'{stage} generation not complete')
    samples=[s for p in cache_paths(args,stage) for s in base.read_batch(p)]
    expected=engine.specs_for(stage,state['source_count'],args.seed)
    by_seed={s.spec.seed:s for s in samples}
    ordered=[by_seed[s.seed] for s in expected]
    if [s.spec for s in ordered]!=expected:raise ValueError('Source specification cache mismatch')
    return ordered


def labels_cache(args,stage):
    protocol=mechanics_fingerprint();result={}
    path=args.work_dir/f'{stage}_labels.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('protocol_fingerprint')==protocol:result[(row['key'],row['sha256_pixels'])]=row
    return result


def prepared_pool(args,stage):
    samples=[s for s in read_sources(args,stage) if not engine.rejection(s)]
    labs=labels_cache(args,stage)
    if any((key(s),digest(s.pixels)) not in labs for s in samples):raise ValueError('FEM labeling incomplete')
    return samples,[labs[(key(s),digest(s.pixels))] for s in samples]


def selected_pool(args,stage,require_review=True):
    manifest=read_json(args.work_dir/f'{stage}_selection.json')
    samples,labels=prepared_pool(args,stage)
    lookup={key(s):(s,l) for s,l in zip(samples,labels)}
    pairs=[lookup[k] for k in manifest['selected_keys']]
    hashes=[digest(s.pixels) for s,l in pairs]
    if hashes!=manifest['selected_hashes']:raise ValueError('Selected maps changed')
    passed,excluded=review(args)
    if require_review and any(h not in passed or h in excluded for h in hashes):raise ValueError('Actual pixel review incomplete')
    return [s for s,l in pairs],[l for s,l in pairs],manifest


def generate(args,stage):
    load_base(args.base)
    if stage=='full':
        _,_,pilot=selected_pool(args,'pilot')
        if pilot.get('visual_review')!='pass' or pilot['selected_count']!=50:raise ValueError('50-image pilot gate not passed')
    state=dict(stage=stage,source_count=args.sources,seed=args.seed,batch_size=args.batch_size,complete=False)
    write_json(args.work_dir/f'{stage}_generation_state.json',state)
    sources=engine.acquire(stage,args.sources,args.work_dir,seed=args.seed,batch_size=args.batch_size)
    valid=[s for s in sources if not engine.rejection(s)]
    state.update(complete=True,valid_sources=len(valid),style_counts=dict(Counter(s.diagnostics['style_family'] for s in valid)),
                 rejections=dict(Counter(engine.rejection(s) for s in sources if engine.rejection(s))))
    write_json(args.work_dir/f'{stage}_generation_state.json',state)
    print(json.dumps(state),flush=True)


def label(args,stage):
    samples=[s for s in read_sources(args,stage) if not engine.rejection(s)]
    labels=label_items([(key(s),s.pixels) for s in samples],args.work_dir/f'{stage}_labels.jsonl',args.workers)
    print(json.dumps(dict(labelled=len(labels),stage=stage)),flush=True)


def stream_labels(args,stage):
    """One writer, reusable CPU workers; generation atomically publishes batches."""
    cached=labels_cache(args,stage);seen=set();pending={};submitted=set();count=0;started=time.monotonic()
    destination=args.work_dir/f'{stage}_labels.jsonl'
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker) as pool, destination.open('a',encoding='utf-8') as out:
        while True:
            for path in cache_paths(args,stage):
                if path in seen:continue
                if len(pending)>=args.workers*4:break
                samples=base.read_batch(path);seen.add(path)
                for sample in samples:
                    pair=(key(sample),digest(sample.pixels))
                    if engine.rejection(sample) or pair in cached or pair in submitted:continue
                    future=pool.submit(solve_item,(key(sample),sample.pixels))
                    pending[future]=pair;submitted.add(pair)
            if pending:
                done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    row=future.result();pair=pending.pop(future)
                    if (row['key'],row['sha256_pixels'])!=pair:raise ValueError('FEM source association failed')
                    out.write(json.dumps(row)+'\n');out.flush();cached[pair]=row;count+=1
                    if count%100==0:print(f'{stage} streaming FEM: {count} new, {time.monotonic()-started:.1f}s',flush=True)
            else:time.sleep(3)
            state=read_json(args.work_dir/f'{stage}_generation_state.json',{})
            if state.get('complete') and not pending and len(seen)==len(cache_paths(args,stage)):
                print(json.dumps(dict(stage=stage,streaming_completed=True,new_labels=count)),flush=True);break


def quotas(total):
    return {family:total//len(engine.STYLE_FAMILIES)+int(i<total%len(engine.STYLE_FAMILIES))
            for i,family in enumerate(engine.STYLE_FAMILIES)}


def select(args,stage):
    fixed,fixed_labels,_,_=load_base(args.base)
    desired=quotas(50 if stage=='pilot' else args.total-BASE_COUNT)
    if stage=='full':
        pilot,plabels,pmanifest=selected_pool(args,'pilot')
        if pmanifest.get('visual_review')!='pass':raise ValueError('Pilot not approved')
        for s in pilot:desired[s.diagnostics['style_family']]-=1
        fixed=list(fixed)+pilot;fixed_labels=list(fixed_labels)+plabels
    pool,labels=prepared_pool(args,stage);passed,excluded=review(args)
    eligible=[i for i,s in enumerate(pool) if digest(s.pixels) not in excluded]
    pool=[pool[i] for i in eligible];labels=[labels[i] for i in eligible]
    chosen,stats,audit=selection.select_additions(fixed,fixed_labels,pool,labels,family_quotas=desired,seed=args.seed)
    selected=[pool[i] for i in chosen];labs=[labels[i] for i in chosen]
    hashes=[digest(s.pixels) for s in selected]
    manifest=dict(stage=stage,selected_count=len(selected),target=sum(desired.values()),family_quotas=desired,
        base_identity=BASE_IDENTITY,fixed_identity=previous.fixed_identity(fixed,fixed_labels),
        selected_keys=[key(s) for s in selected],selected_hashes=hashes,
        selected_seeds=[int(s.spec.seed) for s in selected],
        visual_review='pass' if len(selected)==sum(desired.values()) and all(h in passed for h in hashes) else 'pending',
        statistics=stats,audit=audit,
        selected=[dict(position=i+1,key=key(s),hash=digest(s.pixels),style_family=s.diagnostics['style_family'],
                       reference_style_id=s.diagnostics.get('reference_style_id',''),Ex=l['Ex_app'],Ey=l['Ey_app'],
                       black_fraction=s.black_fraction) for i,(s,l) in enumerate(zip(selected,labs))])
    write_json(args.work_dir/f'{stage}_selection.json',manifest)
    if stage=='pilot':
        contact_sheet(selected,args.work_dir/'pilot_contact_sheet.png',columns=10,tile=224,
            labels=[f"{i+1} {s.diagnostics['style_family']}\nB={s.black_fraction:.1%} Ex={l['Ex_app']:.2f} Ey={l['Ey_app']:.2f}" for i,(s,l) in enumerate(zip(selected,labs))])
    elif any(h not in passed for h in hashes):
        # Pixel-bound source-page review already covers approved selections.
        # Avoid producing a second redundant set of contact sheets.
        make_sheets(args,stage)
    print(json.dumps({k:manifest[k] for k in ('stage','selected_count','target','family_quotas','visual_review')}),flush=True)


def make_sheets(args,stage):
    samples,labels,_=selected_pool(args,stage,require_review=False)
    folder=args.work_dir/'review_sheets';folder.mkdir(exist_ok=True)
    for start in range(0,len(samples),100):
        chosen=samples[start:start+100];labs=labels[start:start+100]
        captions=[f"#{start+i+1} ref{s.diagnostics.get('reference_style_id','')} B={s.black_fraction:.0%}\nEx={l['Ex_app']:.2f} Ey={l['Ey_app']:.2f}" for i,(s,l) in enumerate(zip(chosen,labs))]
        contact_sheet(chosen,folder/f'{stage}_{start//100:03d}.png',columns=10,tile=224,labels=captions)


def record_review(args,stage):
    # This records explicit, already-performed AI inspection; it is not a VLM.
    manifest=read_json(args.work_dir/f'{stage}_selection.json')
    if args.start<1 or args.end>manifest['selected_count'] or args.start>args.end:raise ValueError('Bad review range')
    exclusions={int(v.split(':',1)[0]):v.split(':',1)[1] for v in args.exclude}
    if any(i<args.start or i>args.end for i in exclusions):raise ValueError('Exclude outside inspected range')
    data=read_json(args.work_dir/'reference_visual_review.json',dict(REFERENCE_STYLE_REVIEW))
    passed=set(data.get('passed_hashes',[]));excluded=data.get('excluded_hashes',{})
    for pos in range(args.start,args.end+1):
        h=manifest['selected_hashes'][pos-1]
        if pos in exclusions:excluded[h]=exclusions[pos];passed.discard(h)
        elif h not in excluded:passed.add(h)
    data.update(passed_hashes=sorted(passed),excluded_hashes=excluded,
                method='AI contact-sheet inspection of actual binary maps; ambiguous specimens enlarged; no human review claimed')
    data.setdefault('review_batches',[]).append(dict(stage=stage,start=args.start,end=args.end,
        selection_hashes=manifest['selected_hashes'][args.start-1:args.end],excluded_cells=exclusions))
    write_json(args.work_dir/'reference_visual_review.json',data)
    print(json.dumps(dict(passed=len(passed),excluded=len(excluded))),flush=True)


def approve_pilot(args):
    samples,_,manifest=selected_pool(args,'pilot')
    if len(samples)!=50:raise ValueError('Pilot must contain50')
    manifest['visual_review']='pass';write_json(args.work_dir/'pilot_selection.json',manifest)
    print('50-image pilot approved after exact-pixel review',flush=True)


def new_row(sample,lab,index,args):
    row=previous.new_row(sample,lab,'reference',index,args)
    row.update(method='independent reference-style local diffusion; deterministic style rendering and binary preprocessing; FEM coverage selection',
        pipeline_version=8,generation_stage='reference_style_expansion',generation_master_seed=args.seed,
        candidate_variant_key=key(sample),variant_id=0,source_id=sample.spec.candidate)
    for field,value in sample.diagnostics.items():
        if field not in row and isinstance(value,(str,int,float,bool)):
            row[field]=value
        elif field in ('style_scores','old_style_scores'):
            row[field+'_json']=json.dumps(value,sort_keys=True,separators=(',',':'))
    return row


def export(args):
    if args.output.resolve()==args.base.resolve():raise ValueError('Export cannot overwrite the immutable base')
    fixed,flabels,rows,pixels=load_base(args.base)
    pilot,plabs,pm=selected_pool(args,'pilot');full,labs,fm=selected_pool(args,'full')
    added=pilot+full;alabs=plabs+labs
    if len(added)!=args.total-BASE_COUNT:raise ValueError('Requested image count unavailable; no padding')
    if fm['fixed_identity']!=previous.fixed_identity(list(fixed)+pilot,list(flabels)+plabs):raise ValueError('Fixed base/pilot changed')
    expected=quotas(len(added))
    if Counter(s.diagnostics['style_family'] for s in added)!=expected:raise ValueError('Style quotas differ')
    ids,stats,audit=selection.select_additions(fixed,flabels,added,alabs,family_quotas=expected,seed=args.seed)
    if set(ids)!=set(range(len(added))):raise ValueError('Joint final duplicate/source audit failed')
    write_json(args.work_dir/'final_pair_audit.json',audit)
    args.output.mkdir(parents=True,exist_ok=True);image_dir=args.output/'images';image_dir.mkdir(exist_ok=True)
    expected_names={f'cat_{i:04d}.png' for i in range(args.total)}
    if any(p.name not in expected_names for p in image_dir.glob('*.png')):raise ValueError('Unexpected output PNGs')
    for row in rows:shutil.copy2(args.base/'images'/row['filename'],image_dir/row['filename'])
    for j,(s,l) in enumerate(zip(added,alabs)):
        index=BASE_COUNT+j;base.save_png(s.pixels,image_dir/f'cat_{index:04d}.png');rows.append(new_row(s,l,index,args))
    array=np.concatenate([pixels,np.stack([s.pixels for s in added])]);np.save(args.output/'dataset.npy',array,allow_pickle=False)
    cells=[(min(5,int((float(r['Ex_app'])-1)/1.5)),min(5,int((float(r['Ey_app'])-1)/1.5))) for r in rows]
    counts=Counter(cells);preview=np.random.default_rng(args.seed+991).choice(len(rows),100,replace=False).tolist()
    preview_cells={i:c for c,i in enumerate(preview)}
    for i,(row,cell) in enumerate(zip(rows,cells)):
        row.update(property_bin_x=cell[0],property_bin_y=cell[1],property_bin_count=counts[cell],
                   balanced_sampling_weight=len(rows)/(len(counts)*counts[cell]),preview_cell=preview_cells.get(i,''))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'metadata.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    contact_sheet([SimpleNamespace(pixels=array[i]) for i in preview],args.output/'preview.png',columns=10,tile=224,
        labels=[f"cat_{i:04d} B={float(rows[i]['black_fraction']):.1%}\nEx={float(rows[i]['Ex_app']):.2f} Ey={float(rows[i]['Ey_app']):.2f}" for i in preview])
    write_json(args.work_dir/'release_identity.json',dict(count=len(rows),identity=identity(rows),base_identity=BASE_IDENTITY,
        new_style_counts=dict(Counter(r['style_family'] for r in rows[BASE_COUNT:])),preview_indices=preview))
    write_report(args,rows,audit)
    validate(args)


def write_report(args,rows,audit):
    v=np.array([[float(r['Ex_app']),float(r['Ey_app'])] for r in rows]);f=np.array([float(r['black_fraction']) for r in rows])
    edges=np.arange(1,10.01,.5)
    old=np.histogram2d(v[:BASE_COUNT,0],v[:BASE_COUNT,1],bins=(edges,edges))[0]
    new=np.histogram2d(v[:,0],v[:,1],bins=(edges,edges))[0]
    coarse=np.linspace(1,10,7);before=np.histogram2d(v[:BASE_COUNT,0],v[:BASE_COUNT,1],bins=(coarse,coarse))[0]
    after=np.histogram2d(v[:,0],v[:,1],bins=(coarse,coarse))[0];support=after>0
    cv=lambda a:float(a[support].std()/a[support].mean())
    fractions_before=np.histogram(f[:BASE_COUNT],bins=np.linspace(0,1,21))[0]
    fractions_after=np.histogram(f,bins=np.linspace(0,1,21))[0]
    fraction_cv=lambda counts:float(counts.std()/counts.mean())
    new_cells=int(np.count_nonzero((new>0)&(old==0)))
    unchanged_ranges=bool(np.array_equal(v[:BASE_COUNT].min(axis=0),v.min(axis=0)) and
                          np.array_equal(v[:BASE_COUNT].max(axis=0),v.max(axis=0)))
    more_even=cv(after)<cv(before)
    styles=Counter(r['style_family'] for r in rows[BASE_COUNT:])
    style_text=(f'{len(styles)} 类、每类 {next(iter(styles.values()))} 张'
                if styles and len(set(styles.values()))==1 else json.dumps(styles,sort_keys=True))
    lines=['# Binary cat dataset 质量报告','',
        f'**共 {len(rows):,} 张：原 1,250 张 + 新增 {len(rows)-BASE_COUNT:,} 张。** 原图像素、文件名、seed、Ex/Ey 保持不变；尚未训练模型或划分 training/test。','',
        f'- 全部 PNG 为 224×224、one-bit，重新读取与 `dataset.npy` 一致；数组为 `({len(rows)},224,224)`、`uint8`、值仅 {{0,1}}，无空白图。0=黑色硬固体，1=白色软固体，不是孔洞。',
        f'- 新增参考风格：{style_text}；不增加 05 旧剪影，11 的月亮/星星/外框不进入材料图。每个独立 source 仅保留一张。',
        '- Ex/Ey 均实际 FEM 求解：与原数据相同的 Q4 plane stress，Esoft=1、Ehard=10、两相 ν=0.3，40×40×2 mm 独立试件、0.1% 轴向应变与横向自由边界。Residual、energy/reaction、force balance 及 fixed-fraction 必要界检查通过。','',
        '| 指标 | 原 1,250 张 | 扩展后 |','|---|---:|---:|',
        f'| Ex 范围 | {v[:BASE_COUNT,0].min():.4f}–{v[:BASE_COUNT,0].max():.4f} | {v[:,0].min():.4f}–{v[:,0].max():.4f} |',
        f'| Ey 范围 | {v[:BASE_COUNT,1].min():.4f}–{v[:BASE_COUNT,1].max():.4f} | {v[:,1].min():.4f}–{v[:,1].max():.4f} |',
        f'| 已占据 0.5×0.5 模量格 | {np.count_nonzero(old)} | {np.count_nonzero(new)} |',
        f'| 同一观察支持上的 6×6 格计数 CV | {cv(before):.3f} | {cv(after):.3f} |',
        f'| 黑色面积比例范围 | {f[:BASE_COUNT].min():.2%}–{f[:BASE_COUNT].max():.2%} | {f.min():.2%}–{f.max():.2%} |',
        f'| 黑色比例 20 格计数 CV（每格 5%） | {fraction_cv(fractions_before):.3f} | {fraction_cv(fractions_after):.3f} |','',
        f'本轮新占据 {new_cells} 个 0.5×0.5 模量格；Ex/Ey 全局范围'+('未变。' if unchanged_ranges else '有所扩展。')+
        (' 同一观察支持上的格计数较原数据更均衡。' if more_even else ' 同一观察支持上的格计数没有变得更均衡。'),
        'CV 越低表示这些格的计数越均衡；统计依赖分箱，不证明整个可达域已均匀覆盖。未用背景噪声或填充图案强制面积比例。',
        '',f'**去重：** exact duplicate 和重复 source seed 均为 0。涉及新增的 {audit["audited_new_base_pairs"]+audit["audited_new_new_pairs"]:,} 对，剩余 mechanical-near={audit["mechanical_near_conflicts_remaining"]}、strict-visual-near={audit["strict_visual_conflicts_remaining"]}。',
        'Mechanical-near 规则：同 polarity、黑色比例差≤0.025、Ex/Ey 的 L∞ 距离≤0.12，并满足 silhouette IoU≥0.91，或 IoU≥0.85 且 pHash 距离≤6。Strict-visual 规则独立于模量/反相：归一化 actual-ink 与填孔 silhouette 的 IoU 均≥0.97，且 pHash≤4；均检查一致的镜像比较。',
        '',f'**可识别性：** 先完成 50 张 pilot，再扩展。全部保留的 {len(rows)-BASE_COUNT:,} 张新图经过 actual-pixel AI contact-sheet review，疑点放大检查，并通过 semantic/style screening。没有 human inspection 或完美可识别性声明；requested pose/view/style 是生成意图，风格可重叠。',
        '', '**局限：** 增加风格不保证扩大模量边界；稀疏区和密度不均仍可能存在。细纹理、像素角点的可制造性及 3D、非线性、失效、实验响应未验证；小代数残差不等于小 continuum error。']
    review_data=read_json(args.work_dir/'reference_visual_review.json',REFERENCE_STYLE_REVIEW)
    base_audit=review_data.get('fixed_base_pair_audit',{})
    if base_audit:
        if base_audit.get('base_identity')!=BASE_IDENTITY:raise ValueError('Base pair-audit identity differs')
        pairs=base_audit['all_base_pairs']+audit['audited_new_base_pairs']+audit['audited_new_new_pairs']
        if pairs!=len(rows)*(len(rows)-1)//2:raise ValueError('Incomplete pair accounting')
        mechanical=base_audit['mechanical_near_pairs']+audit['mechanical_near_conflicts_remaining']
        visual=base_audit['strict_visual_near_pairs']+audit['strict_visual_conflicts_remaining']
        lines+=['',f'合并原数据的 {base_audit["all_base_pairs"]:,} 对复查后，上述规则覆盖全数据集 {pairs:,} 对（以必要条件精确剪枝）；mechanical-near={mechanical}，strict-visual-near={visual}。这不是任意人类相似性标准下的唯一性保证。']
    refinement=read_json(args.work_dir/'refinement.json')
    if refinement:lines+=['',f"数值抽查：{refinement['count']} 张在保持材料几何不变的 224→448 网格细化中，最大模量变化 {refinement['maximum_relative_change']:.3%}；仅为抽样证据，并非 continuum error 上界。"]
    reproduction=read_json(args.work_dir/'reproduction.json')
    if reproduction:lines+=['',f"复现抽查：本机 {reproduction['batches']} 个原 batch、{reproduction['checked_sources']} 个 source，覆盖全部新风格；binary pixels/source masks 全部一致={reproduction['all_exact']}。不保证跨平台 bitwise 相同。"]
    (args.output/'quality_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def validate(args):
    folder=args.output;rows=read_rows(folder);array=np.load(folder/'dataset.npy',mmap_mode='r',allow_pickle=False);n=len(rows)
    if n!=args.total:raise ValueError(f'Expected {args.total} specimens, found {n}')
    if array.shape!=(n,224,224) or array.dtype!=np.uint8:raise ValueError('Wrong array dimensions/dtype')
    if identity(rows[:BASE_COUNT])!=BASE_IDENTITY:raise ValueError('Original1250 metadata changed')
    names=[f'cat_{i:04d}.png' for i in range(n)]
    if [r['filename'] for r in rows]!=names or sorted(p.name for p in (folder/'images').glob('*.png'))!=names:raise ValueError('Filename ordering/count')
    protocol=mechanics_fingerprint();hashes=[];seeds=[];cells=Counter((int(r['property_bin_x']),int(r['property_bin_y'])) for r in rows)
    for i,r in enumerate(rows):
        p=base.load_png(folder/'images'/r['filename']);h=digest(p);hashes.append(h);seeds.append(r['seed'])
        if p.shape!=(224,224) or set(np.unique(p))!={0,1} or not np.array_equal(p,array[i]) or h!=r['sha256_pixels']:raise ValueError(f'Pixel validation:{i}')
        fraction=float((p==0).mean());x=float(r['Ex_app']);y=float(r['Ey_app'])
        if abs(fraction-float(r['black_fraction']))>1e-12 or int((p==0).sum())!=int(r['black_pixels']):raise ValueError('Fraction metadata')
        if min(x,y)<1/(1-.9*fraction)-1e-6 or max(x,y)>1+9*fraction+1e-6:raise ValueError('Necessary mechanical bound')
        if r['solver_protocol_sha256']!=protocol:raise ValueError('FEM protocol')
        for k,v in dict(E_soft=1,E_hard=10,nu_soft=.3,nu_hard=.3,specimen_width_mm=40,specimen_height_mm=40,specimen_thickness_mm=2,nominal_strain=.001).items():
            if float(r[k])!=v:raise ValueError('Physical parameters')
        for axis in ('x','y'):
            if float(r[f'E{axis}_over_Es'])!=float(r[f'E{axis}_app']):raise ValueError('Condition alias')
            for prefix,limit in [('residual',1e-7),('energy_relative_error',1e-6),('force_balance_relative_error',1e-7)]:
                value=float(r[f'{prefix}_{axis}'])
                if not np.isfinite(value) or not 0<=value<=limit:raise ValueError('Solver diagnostics')
        if r['validation']!='pass' or r['ai_visual_review']!='pass' or r['fem_valid']!='True':raise ValueError('QC incomplete')
        if i>=BASE_COUNT:
            if r.get('style_family') not in engine.STYLE_FAMILIES:raise ValueError('Excluded/unknown new style')
            reference_name,reference_hash=engine.REFERENCE_FILES[r['style_family']]
            if r.get('reference_filename')!=reference_name or r.get('reference_sha256')!=reference_hash:raise ValueError('Reference provenance')
            if not .05<=fraction<=.97:raise ValueError('New-map phase fraction range')
        grid=int(r.get('pixel_art_grid') or 0)
        if grid:
            block=224//grid
            if 224%grid or not np.array_equal(p,np.repeat(np.repeat(p[::block,::block],block,0),block,1)):raise ValueError('Pixel grid')
        cell=(min(5,int((x-1)/1.5)),min(5,int((y-1)/1.5)))
        if cell!=(int(r['property_bin_x']),int(r['property_bin_y'])) or int(r['property_bin_count'])!=cells[cell]:raise ValueError('Property bins')
        if not np.isclose(float(r['balanced_sampling_weight']),n/(len(cells)*cells[cell]),rtol=1e-12):raise ValueError('Sampling weights')
    if len(set(hashes))!=n or len(set(seeds))!=n:raise ValueError('Pixel/source duplicates')
    members=[(int(r['preview_cell']),i) for i,r in enumerate(rows) if r['preview_cell']!='']
    if sorted(c for c,i in members)!=list(range(100)):raise ValueError('Preview membership')
    with Image.open(folder/'preview.png') as im:
        if im.size!=(2306,2646):raise ValueError('Preview dimensions')
        for cell,i in members:
            x=6+(cell%10)*230;y=6+(cell//10)*264
            if not np.array_equal(np.asarray(im.crop((x,y,x+224,y+224)),dtype=np.uint8),array[i]):raise ValueError('Preview content')
    print(json.dumps(dict(valid_images=n,shape=list(array.shape),dtype='uint8',values=[0,1],original1250_preserved=True,exact_duplicates=0)),flush=True)


def extract_base(args):
    if args.output.resolve()==args.base.resolve():raise ValueError('Extraction output must be separate')
    rows=read_rows(args.base)[:BASE_COUNT];pixels=np.load(args.base/'dataset.npy',allow_pickle=False)[:BASE_COUNT]
    if identity(rows)!=BASE_IDENTITY or any(digest(p)!=r['sha256_pixels'] for p,r in zip(pixels,rows)):raise ValueError('Invalid original1250 prefix')
    (args.output/'images').mkdir(parents=True,exist_ok=True)
    if any(p.name not in {r['filename'] for r in rows} for p in (args.output/'images').glob('*.png')):raise ValueError('Unexpected extraction PNGs')
    for r in rows:shutil.copy2(args.base/'images'/r['filename'],args.output/'images'/r['filename'])
    np.save(args.output/'dataset.npy',pixels,allow_pickle=False)
    with (args.output/'metadata.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Original1250 pixels/labels extracted',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['generate-pilot','generate-full','label-pilot','label-full','stream-full','select-pilot','select-full','review-pilot','review-full','approve-pilot','sheets-full','export','validate','extract-base'])
    p.add_argument('--work-dir',type=Path,default=Path(tempfile.gettempdir())/'binary-cat-reference-expansion')
    p.add_argument('--base',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--sources',type=int,default=144);p.add_argument('--total',type=int,default=8000)
    p.add_argument('--seed',type=int,default=20261114);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=50);p.add_argument('--exclude',action='append',default=[])
    args=p.parse_args();args.work_dir.mkdir(parents=True,exist_ok=True)
    args.base=args.base or args.work_dir/'base1250';args.output=args.output or args.work_dir/'release'
    if args.mode.startswith('generate-'):generate(args,args.mode.split('-')[1])
    elif args.mode.startswith('label-'):label(args,args.mode.split('-')[1])
    elif args.mode.startswith('stream-'):stream_labels(args,args.mode.split('-')[1])
    elif args.mode.startswith('select-'):select(args,args.mode.split('-')[1])
    elif args.mode.startswith('review-'):record_review(args,args.mode.split('-')[1])
    elif args.mode=='approve-pilot':approve_pilot(args)
    elif args.mode=='sheets-full':make_sheets(args,'full')
    elif args.mode=='export':export(args)
    elif args.mode=='extract-base':extract_base(args)
    else:validate(args)
    return args


if __name__=='__main__':main()
