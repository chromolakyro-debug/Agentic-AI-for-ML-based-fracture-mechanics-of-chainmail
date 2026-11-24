"""Build a compact, evidence-first AI review package from the frozen CFM v0 run."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=HERE/'run'
PREFIX='cfm_v0_ai_review/'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


VERIFY_SCRIPT=r'''"""Verify this AI-review ZIP with Python's standard library; no GPU needed."""
import argparse
import hashlib
import json
import zipfile
from pathlib import PurePosixPath

def verify(path):
    prefix='cfm_v0_ai_review/'
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        assert len(names)==len(set(names))
        assert all(n.startswith(prefix) and '..' not in PurePosixPath(n).parts and '\\' not in n for n in names)
        raw=z.read(prefix+'AUDIT_MANIFEST.json');m=json.loads(raw)
        checks={p:h for h,p in (line.split('  ',1) for line in z.read(prefix+'SHA256SUMS.txt').decode('utf-8').splitlines())}
        assert checks['AUDIT_MANIFEST.json']==hashlib.sha256(raw).hexdigest()
        expected={prefix+r['path'] for r in m['files']}|{prefix+'AUDIT_MANIFEST.json',prefix+'SHA256SUMS.txt'}
        assert set(names)==expected
        assert set(checks)=={r['path'] for r in m['files']}|{'AUDIT_MANIFEST.json'}
        total=0
        for r in m['files']:
            h=hashlib.sha256();size=0
            with z.open(prefix+r['path']) as f:
                for chunk in iter(lambda:f.read(2**20),b''):
                    h.update(chunk);size+=len(chunk)
            assert h.hexdigest()==r['sha256']==checks[r['path']],r['path']
            assert size==r['bytes'],r['path']
            total+=size
    return {'files':len(m['files']),'uncompressed_payload_bytes':total,'sha256_and_crc_pass':True,'exact_inventory_pass':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args()
    print(json.dumps(verify(a.archive),indent=2))
'''


def build(stage,destination):
    import numpy as np
    from PIL import Image
    if stage.exists() or (destination is not None and destination.exists()):
        raise FileExistsError('Use new paths; existing evidence packages are preserved.')
    protocol=read(RUN/'protocol.json')
    assert digest(RUN/'protocol.json')==read(RUN/'frozen_protocol_sha256.json')['sha256']
    assert digest(ROOT/'cat_binary_dataset'/'metadata.csv')==protocol['metadata_sha256']
    assert digest(ROOT/'cat_binary_dataset'/'dataset.npy')==protocol['data_sha256']
    for name,expected in read(RUN/'source_sha256_final.json').items():
        assert digest(ROOT/name.replace('\\','/'))==expected,name
    assert read(RUN/'final_audit.json')['status']=='complete'
    stage.mkdir(parents=True)
    provenance={}
    def copy(source,relative=None):
        source=Path(source);relative=relative or source.relative_to(ROOT).as_posix()
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        provenance[relative]=dict(source=source.relative_to(ROOT).as_posix(),source_sha256=digest(source),operation='byte-for-byte copy')
    source_names=['common','prepare','model','train','generate','fem','diagnostics','metrics','plots',
                  'check_core','check_resume','audit_final','finish','report','record_visual_review']
    for name in source_names:copy(HERE/(name+'.py'))
    for name in ['mechanics','evaluate_mechanics','mechanics_checks','mechanics_selection','reference_style_selection','augmentation_selection']:
        copy(ROOT/(name+'.py'))
    for name in ['requirements.txt','run_commands.ps1','support_and_targets.png','controllability.png','generated_examples.png']:
        copy(HERE/name)
    original_report=(HERE/'REPORT.md').read_text(encoding='utf-8')
    reviewed_report=original_report.replace('[checkpoint](run/checkpoints/best.pt)',
                           '[checkpoint 身份记录（权重未附）](run/final/manifest.json)')
    reviewed_report+='\n审阅包说明：本文件为原报告的审阅副本，仅将未附 checkpoint 的链接改为其身份记录。实验数字和结论未修改。完整训练/生成复现仍需原始图像和模型权重；见 [START_HERE.md](../START_HERE.md)。\n'
    (stage/'cfm_v0'/'REPORT.md').write_text(reviewed_report,encoding='utf-8')
    provenance['cfm_v0/REPORT.md']=dict(source='cfm_v0/REPORT.md',source_sha256=digest(HERE/'REPORT.md'),
              operation='Replace omitted checkpoint link with its manifest; append explicit review-copy note; numerical content unchanged')
    run_files=['protocol.json','frozen_protocol_sha256.json','train_config.json','splits.json','data.npz',
               'data_audit.json','core_checks.json','smoke_fem.json','smoke_training.json','resume_check.json',
               'environment.json','training_summary.json','training.jsonl','test_once.json','retrieval.json',
               'coverage_diagnostics.json','development_progress.json','final_audit.json','source_sha256_final.json',
               'completion_commands.json','evaluation_rules.json','evaluation_completed.json']
    for name in run_files:copy(RUN/name)
    final_files=['binary.npy','manifest.json','samples.csv','samples.json','target_summary.csv','metrics.json',
                 'image_neighbors.json','fem.jsonl','fem_summary.json','visual_inspection.json','visual_review_manifest.json',
                 'visual_review_fixed_seeds.png','figure_examples.json','threshold_margin_diagnostics.json','interpretation.json']
    for name in final_files:copy(RUN/'final'/name)
    for checkpoint in ['development_001000','development_010000']:
        for name in ['binary.npy','manifest.json','fem.jsonl','fem_summary.json','visual_review_fixed_seeds.png','visual_inspection.json']:
            copy(RUN/checkpoint/name)
    copy(ROOT/'cat_binary_dataset'/'README.md','context/DATASET_README.md')
    copy(ROOT/'cat_binary_dataset'/'quality_report.md','context/DATASET_QUALITY_REPORT.md')
    copy(Path(__file__),'audit_tools/package_ai_review.py')
    (stage/'VERIFY_AI_ARCHIVE.py').write_text(VERIFY_SCRIPT,encoding='utf-8')

    metadata=list(csv.DictReader((ROOT/'cat_binary_dataset'/'metadata.csv').open(encoding='utf-8-sig',newline='')))
    d=np.load(RUN/'data.npz');assert len(metadata)==len(d['sa'])==8000
    split=np.full(8000,'',dtype=object)
    for key in ['train','validation','test']:split[d[key]]=key
    assert np.all(split!='')
    weights={int(i):float(w) for i,w in zip(d['train'],d['train_weights'])}
    train_density={int(i):float(r) for i,r in zip(d['train'],d['train_r20'])}
    val_support={int(i):float(r) for i,r in zip(d['validation'],d['validation_r20'])}
    props=[]
    retained=['filename','sha256_pixels','source_id','source_seed','seed','generation_stage','style_family',
              'reference_style_id','reference_style_name','requested_style','requested_pose','requested_view','portrait',
              'background_value','black_fraction','Ex_app','Ey_app','E_soft','E_hard']
    for i,row in enumerate(metadata):
        assert row['filename']==f'cat_{i:04d}.png'
        assert np.allclose([float(row['Ex_app']),float(row['Ey_app'])],d['moduli'][i],atol=1e-12,rtol=0)
        record={'index':i,'split':split[i],**{k:row.get(k,'') for k in retained},
                's':float(d['sa'][i,0]),'a':float(d['sa'][i,1]),
                'train_sampling_weight':weights.get(i,''),'train_r20_leave_self_out':train_density.get(i,''),
                'validation_to_training_r20':val_support.get(i,'')}
        props.append(record)
    slim=stage/'context'/'dataset_properties_8000.csv'
    with slim.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(props[0]));writer.writeheader();writer.writerows(props)

    samples=read(RUN/'final'/'samples.json');by_key={r['key']:r for r in samples}
    neighbors={r['key']:r for r in read(RUN/'final'/'image_neighbors.json')['results']}
    retrieval={r['target_id']:r for r in read(RUN/'retrieval.json')['results']}
    targets={t['target_id']:t for t in protocol['final_targets']}
    final_manifest=read(RUN/'final'/'manifest.json')
    image_positions={r['key']:r['index'] for r in final_manifest['samples']}
    generated=np.load(RUN/'final'/'binary.npy',mmap_mode='r')
    training=np.load(ROOT/'cat_binary_dataset'/'dataset.npy',mmap_mode='r')
    comparison_keys=['T00_seed730000','T05_seed730000','T10_seed730000','T12_seed730000','T16_seed730000','T19_seed730001']
    comparisons=[];train_ids=set(d['train'].tolist());used_training=set()
    for key in comparison_keys:
        sample=by_key[key];r=retrieval[sample['target_id']];n=neighbors[key]
        prop_id=r['training_index'];visual_id=n['shape_nearest_training_index']
        assert prop_id in train_ids and visual_id in train_ids
        for idx in [prop_id,visual_id]:
            assert hashlib.sha256(training[idx].tobytes()).hexdigest()==metadata[idx]['sha256_pixels']
            used_training.add(idx)
        comparisons.append(dict(key=key,generated_index=image_positions[key],target_id=sample['target_id'],
                 seed=sample['seed'],target_s=sample['s'],target_a=sample['a'],fem_s=sample['s_hat'],fem_a=sample['a_hat'],
                 generated_pixel_sha256=sample['sha256_pixels'],hit_joint=sample['hit_joint'],density=targets[sample['target_id']]['density'],
                 property_retrieval_training_index=prop_id,property_retrieval_s=r['s'],property_retrieval_a=r['a'],
                 property_retrieval_error_scaled_l2=r['error_scaled_l2'],
                 visual_neighbor_training_index=visual_id,visual_neighbor_s=float(d['sa'][visual_id,0]),
                 visual_neighbor_a=float(d['sa'][visual_id,1]),visual_neighbor_shape_pooled_rms=n['shape_pooled_rms'],
                 property_retrieval_pixel_sha256=metadata[prop_id]['sha256_pixels'],
                 visual_neighbor_pixel_sha256=metadata[visual_id]['sha256_pixels']))
    for idx in sorted(used_training):
        copy(ROOT/'cat_binary_dataset'/'images'/metadata[idx]['filename'],
             'context/reference_training_images/'+metadata[idx]['filename'])
    comparison=stage/'context'/'retrieval_comparison.json'
    write_json(comparison,dict(selection='Post-evaluation illustrative subset of the pre-fixed first-two seeds; covers density/stiffness regimes and known failures; not an unbiased quality-rate sample',
                property_metric='Training-only Euclidean distance in (s/log(1.05), a/log(1.10))',
                visual_metric='Training-only minimum reflection-aware bbox-normalized pooled actual-ink RMS, reused from image_neighbors.json; not a human semantic similarity score',
                display='Original 224x224 pixels and original material polarity, no geometry transformations',cases=comparisons))
    create_comparison(stage,comparisons,generated,training)

    write_docs(stage)
    observed=read(RUN/'final'/'metrics.json')['overall']
    assert len(samples)==640 and sum(r['hit_joint'] for r in samples)==244
    assert read(RUN/'final'/'visual_inspection.json')['manifest_sha256']==digest(RUN/'final'/'manifest.json')
    assert len(read(RUN/'final'/'visual_inspection.json')['samples'])==40
    for i,row in enumerate(final_manifest['samples']):
        assert hashlib.sha256(generated[i].tobytes()).hexdigest()==row['sha256_pixels']
    for path in (stage/'context'/'reference_training_images').glob('*.png'):
        idx=int(path.stem.split('_')[1]);pixels=np.asarray(Image.open(path).convert('L'))//255
        assert np.array_equal(pixels,training[idx])
    checks=dict(input_metadata_and_dataset_frozen_hashes_verified=True,original_source_hashes_verified=True,
             slim_rows=8000,split_counts={k:int(np.sum(split==k)) for k in ['train','validation','test']},
             slim_properties_match_original_metadata_and_frozen_arrays=True,final_binary_count=640,
             final_binary_pixel_hashes_match=True,all_final_fem_valid=all(r['fem_valid'] for r in samples),
             joint_hit_count=244,retrieval_and_visual_comparison_training_only=True,
             comparison_cases=len(comparisons),reference_training_images=len(used_training),
             reference_png_pixels_match_original_dataset=True,fixed_visual_review_count=40,
             no_model_weights_or_environment=True,no_full_original_images_or_array=True)
    write_json(stage/'PACKAGE_CHECKS.json',checks)
    write_json(stage/'SOURCE_PROVENANCE.json',dict(copies=provenance,
                derived=dict(slim_metadata='Selected columns joined by original index to frozen splits/conditions/weights; no labels re-estimated',
                    comparison='Six illustrative fixed-seed cases; original training pixels; existing FEM labels and existing exact training-neighbor indices',
                    documentation='New AI reading guide and research questions, facts drawn from frozen artifacts; future budget undecided')))
    print('Review files prepared:',stage,flush=True)
    if destination is not None:make_zip(stage,destination)


def create_comparison(stage,cases,generated,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,axes=plt.subplots(6,3,figsize=(10.5,20))
    for row,case in enumerate(cases):
        arrays=[generated[case['generated_index']],training[case['property_retrieval_training_index']],training[case['visual_neighbor_training_index']]]
        titles=[f"{case['key']} | {'HIT' if case['hit_joint'] else 'MISS'}",
                f"cat_{case['property_retrieval_training_index']:04d} | property NN",
                f"cat_{case['visual_neighbor_training_index']:04d} | visual NN"]
        labels=[f"target ({case['target_s']:.3f}, {case['target_a']:.3f})\nFEM ({case['fem_s']:.3f}, {case['fem_a']:.3f}); {case['density']}",
                f"FEM ({case['property_retrieval_s']:.3f}, {case['property_retrieval_a']:.3f})\nscaled property distance {case['property_retrieval_error_scaled_l2']:.3f}",
                f"FEM ({case['visual_neighbor_s']:.3f}, {case['visual_neighbor_a']:.3f})\nnormalized shape RMS {case['visual_neighbor_shape_pooled_rms']:.3f}"]
        for ax,arr,title,label in zip(axes[row],arrays,titles,labels):
            ax.imshow(arr,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=9,pad=8);ax.set_xlabel(label,fontsize=9,labelpad=5)
            for spine in ax.spines.values():spine.set_color('#dddddd')
    fig.suptitle('Generated map             Training property retrieval             Training visual neighbor\n'
                 'All pairs are (s, a). Original binary pixels: black = hard, white = soft.\n'
                 'Illustrative cases selected after evaluation; aggregate metrics use all 640 generated maps.',fontsize=11,y=.992)
    fig.subplots_adjust(left=.04,right=.98,top=.94,bottom=.035,hspace=.56,wspace=.12)
    fig.savefig(stage/'cfm_v0'/'retrieval_comparison.png',dpi=150,facecolor='white');plt.close(fig)


def write_docs(stage):
    (stage/'START_HERE.md').write_text('''# 给 AI 的 CFM v0 审阅包

目的：审查已完成实验的证据，讨论下一步最有价值的小规模验证。先读本文件和 NEXT_STEP_BRIEF.md，再看报告、图和逐样本数据；不要仅凭报告中的解释作因果判断。

建议阅读顺序：

1. [下一步讨论任务](NEXT_STEP_BRIEF.md) 与 [实测报告](cfm_v0/REPORT.md)。
2. [support 与目标](cfm_v0/support_and_targets.png)、[controllability](cfm_v0/controllability.png)、[生成示例](cfm_v0/generated_examples.png)。
3. [新增 retrieval/视觉近邻对照](cfm_v0/retrieval_comparison.png) 与 [固定 40 张未按 FEM 筛选的图](cfm_v0/run/final/visual_review_fixed_seeds.png)。
4. [20 目标汇总](cfm_v0/run/final/target_summary.csv)、[全部 640 样本](cfm_v0/run/final/samples.csv)、[配对扫描统计](cfm_v0/run/final/metrics.json)。
5. [8000 条精简属性与 split](context/dataset_properties_8000.csv)、[数据字段说明](DATA_FIELDS.md)，按需检查代码与训练日志。

已确认事实：20,000 steps / 33.42 分钟，640 张实际 binary FEM 全部有效；联合命中 244/640=38.125%。Training retrieval 命中 20/20 个目标；CFM 经 FEM 筛选后的 best-of-32 也为 20/20，不能代替随机生成表现。固定 40 张的 AI 像素检查为 20 张基本连贯、19 张失真、1 张难辨，只有 5 张同时符合基本连贯猫形与属性命中；这些是主观 AI 判断，不是人工 ground truth。

重点警惕：固定负 a 时平均交叉斜率接近 0，但逐 seed 的 a_hat 扫描极差中位数仍为 0.226，存在 U 形漂移。不要把平均斜率或 velocity MSE 当成独立控制、视觉质量的充分证据。

本包保留完整 final binary.npy（640×224×224，uint8）和 FEM diagnostics，支持重新分析结果；仅附对照图用到的少量 training PNG，不含完整原始 8000 图像、模型权重、optimizer 状态、.venv 或大缓存。核心代码保持原样供审查，完整重新训练/采样还需要原始数据及环境，直接执行全部原始脚本不保证在精简包中具备所需输入。

数据数值、原始图像和实验代码没有为审阅作改动。REPORT.md 仅调整了未附 checkpoint 的链接并加上说明，原报告 SHA256 保留在 SOURCE_PROVENANCE.json。新的近邻对照是事后说明性选择，不能用于估计总体图像质量；其来源、索引和像素 hash 见 context/retrieval_comparison.json。

校验：解压后运行 `python VERIFY_AI_ARCHIVE.py <ZIP路径>`；只用 Python 标准库，逐文件验证 SHA256、CRC、字节数和精确清单。AUDIT_MANIFEST.json 记录全部资料，SHA256SUMS.txt 同时校验资料与 manifest；ZIP 外的 .sha256 校验整个压缩包。PACKAGE_CHECKS.json 记录打包前的数据一致性核对。
''',encoding='utf-8')
    (stage/'NEXT_STEP_BRIEF.md').write_text('''# 下一步讨论任务

请基于实际数据、图像和代码，分析 CFM v0 的不足，给出最多三个有优先级的小规模验证实验。请先核查报告结论是否被证据支持；不要默认接受已有 AI 的视觉评分或因果解释。

目标：给定总体刚度 s 与各向异性 a，生成视觉上像猫的 binary material maps，并用真实 FEM 验证两项属性是否可独立控制。0 为黑色硬材料、1 为白色软材料，Esoft=1、Ehard=10。s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]，a=log(Ex/Ey)。必须保持原 224×224 几何、材料含义和 FEM 边界条件的一致性。

已完成：固定 8000 张、6400/800/800 split；training-only 标准化、采样权重与 support density；Gaussian-to-image linear CFM，单通道 FiLM conditional U-Net，11,529,633 参数、batch 16、20000 steps。Validation velocity MSE 选择 EMA；test 只在训练结束后评价。Final targets、阈值与 32 个配对 seeds 在生成前固定。所有 640 张终态图按阈值 0 二值化后运行原 FEM。没有 surrogate、augmentation 或额外数据扩充。

请优先回答：

1. s 的响应为何接近 1，而 a 在不同刚度区间仅约 0.60–0.87？固定负 a 时的 U 形漂移、固定正 a 时的系统性下降，各说明了什么？区分系统偏差、seed 离散性和交叉耦合。
2. 密集区域联合命中 58.6%、稀疏区域 33.1%，以及 T16/T17 各 3/32 的失败，哪些可能与数据覆盖、style/材料比例混杂相关？不能仅由相关性断言因果。
3. 视觉质量与力学命中为何脱节？比较固定 seeds、属性 retrieval 与视觉近邻。像素不同、描述子不重复不能证明有用的新结构。
4. 现有证据更支持训练不足、条件利用不足、模型容量、采样积分、数据覆盖还是二值化？把“已证实”“可能”“无法判断”分开。619 张在阈值附近 (−0.1,0.1) 没有终态像素，其中 381 张仍未命中；不要忽略这一证据，也不要把它扩大解释为排除了所有二值化/离散化问题。

下一轮计算预算、优先级和是否允许扩数据尚未由用户确定。本轮没有授权扩数据、引入 surrogate 或新方法。请先优先设计不扩数据、不新增 surrogate 的最小验证；若认为需要放宽约束，请说明理由及代价，不要把它写成已获批准的动作。

建议输出：先给不超过五条证据判断，然后最多三个按优先级排列的实验。每个实验说明要排除的假设、与当前 baseline 相比具体改变什么、保持哪些条件可比、所需 GPU/FEM 预算（估算注明依据）、development panel 的预设成功/停止标准，以及不同结果如何影响下一步选择。避免无上限延长训练或大规模搜索。

所有现有 final-panel 结果已经看过。若据此开发下一版，必须把它标为历史 benchmark，不能再称全新盲测；checkpoint 选择和参数调优仍用 development/validation，新的最终检验条件应事先固定。Test split 不应被反复用于选择方案。
''',encoding='utf-8')
    (stage/'DATA_FIELDS.md').write_text('''# 数据与证据字段

`context/dataset_properties_8000.csv`：index 是原 dataset.npy 的第 0–7999 张，并对应 cat_0000.png–cat_7999.png 及 metadata 行序。Ex_app/Ey_app 是原 FEM apparent moduli；s/a 来自冻结的 data.npz。split 为 train/validation/test。black_fraction 为硬材料比例；background_value 只是原始生成时的视觉背景极性，材料映射始终为 0=硬、1=软。

style_family、reference_style_*、requested_style/pose/view、portrait 来自原生成 metadata，可能为空且表示生成标签/意图，不是人工确认的类别。旧数据没有 reference style 时不要把空值擅自解释成某一种风格。64-bit seed 与 source_seed 保留十进制字符串，不经过浮点转换；prompt 等长文本已省略。sha256_pixels 是原 uint8 0/1 数组的 SHA256。

train_sampling_weight 与 train_r20_leave_self_out 只在 train 行有值；后者排除自身，以第 20 个邻居衡量局部密度。validation_to_training_r20 仅在 validation 行有值；其第 95 分位给出固定 support threshold。Test 没有用于计算条件标准化、权重或阈值。具体均值、标准差、分箱和逐行统计保留在 cfm_v0/run/data.npz 与 protocol.json。

`final/samples.csv` 与 `samples.json` 是同一 640 样本的可读/完整视图：key=target+seed，s/a 为目标，s_hat/a_hat、Ex_app/Ey_app 为最终 binary map 的真实 FEM；delta 为 actual−target。hit_joint 采用固定 |delta_s|≤log(1.05)、|delta_a|≤log(1.10)。原始残差、能量/反力检查、耗时及协议 hash 在 fem.jsonl。manifest.json 绑定数组 index、noise seed、checkpoint hash 与 binary pixel hash。binary.npy 顺序与 manifest 一致。

`target_summary.csv` 汇总全部 32 个 seeds 的绝对误差 median/P90、bias、标准差、命中率和扫描起点的配对固定属性漂移。`metrics.json` 还包含每个 seed 的扫描斜率、逐 seed 固定属性极差和 bootstrap CI。斜率接近 0 可能掩盖非线性漂移，须结合原始样本与图查看。

`retrieval.json` 只在 6400 个 train 候选中按容差归一化属性 L2 检索。`image_neighbors.json` 的像素最近邻和 normalized shape 最近邻是不同指标；视觉 descriptor 允许 bbox 归一化和反射，仅用于相似度诊断，没有把这些变换后的图送入 FEM。strict visual predicate 的零匹配不等于证明有意义的新结构。

新增 retrieval_comparison 图的六个样本来自已固定的前两个 seeds，事后选择覆盖 dense/sparse、多个刚度水平、成功与失败；详细选择与索引见 context/retrieval_comparison.json。所有对照 training 图均属于 train split，展示原像素和原标签。图中第三列是描述子最近邻，不是人工语义判断。

`visual_inspection.json` 的 40 个主观分类均绑定实际 pixel hash；这是 assistant 实际查看，不是 human ground truth。最初 generated_examples 图末四张为明确标注的 FEM 筛选说明例，不能用它们替代全部 640 样本的统计。
''',encoding='utf-8')


def make_zip(stage,destination):
    if destination.exists():raise FileExistsError(destination)
    files=sorted(p for p in stage.rglob('*') if p.is_file() and p.name not in ['AUDIT_MANIFEST.json','SHA256SUMS.txt'])
    assert not any(p.suffix=='.pt' or '.venv' in p.parts or p.name=='training_descriptors.npz' for p in files)
    rows=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
    manifest=dict(format_version=1,package='CFM v0 AI review',created_utc=datetime.now(timezone.utc).isoformat(),
           purpose='Evidence for discussing next experiments; not a complete training recovery environment',
           omitted=['model/optimizer checkpoints','full original 8000 image PNGs and dataset.npy','full metadata prompts',
                    'Python environment','training descriptor cache','duplicate per-target NPZ caches'],files=rows)
    write_json(stage/'AUDIT_MANIFEST.json',manifest)
    sums=''.join(r['sha256']+'  '+r['path']+'\n' for r in rows)
    sums+=digest(stage/'AUDIT_MANIFEST.json')+'  AUDIT_MANIFEST.json\n'
    (stage/'SHA256SUMS.txt').write_text(sums,encoding='utf-8')
    temporary=destination.with_suffix(destination.suffix+'.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for path in files+[stage/'AUDIT_MANIFEST.json',stage/'SHA256SUMS.txt']:
            z.write(path,PREFIX+path.relative_to(stage).as_posix())
    namespace={'__name__':'ai_archive_validation'}
    exec(VERIFY_SCRIPT,namespace)
    validation=namespace['verify'](temporary)
    temporary.replace(destination)
    sha=digest(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(sha+'  '+destination.name+'\n',encoding='utf-8')
    validation.update(archive=str(destination),zip_bytes=destination.stat().st_size,zip_sha256=sha)
    write_json(HERE/'ai_review_package_validation.json',validation)
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,default=HERE/'ai_review_package')
    parser.add_argument('--output',type=Path,default=ROOT/'cfm_v0_ai_review_2026-11-14.zip')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--zip-only',action='store_true')
    args=parser.parse_args()
    if args.zip_only:make_zip(args.stage.resolve(),args.output.resolve())
    else:build(args.stage.resolve(),None if args.prepare_only else args.output.resolve())
