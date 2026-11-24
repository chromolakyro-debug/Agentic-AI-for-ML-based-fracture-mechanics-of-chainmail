"""Package readable source modules into a standalone, reproducible release script.

Virtual modules retain independent namespaces and linecache source text, so
Windows multiprocessing and inspect-based FEM protocol fingerprints work.
"""
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parent
WORK=Path("C:/Users/chrom/AppData/Local/Temp/binary-cat-augmentation")
MODULES=("candidate_generation","mechanics","mechanics_selection","evaluate_mechanics",
         "release_workflow","targeted_candidates","style_candidates","augmentation_selection","augmentation_workflow")


def reference_constants(path):
    wanted={"REFERENCE_REVIEW","REFERENCE_BASELINE","REFERENCE_SOLVER_CHECKS"}
    result={}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in wanted:
                    result[target.id]=ast.literal_eval(node.value)
    return result


def build(destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_readme(destination.parent)
    sources={name:(ROOT/f"{name}.py").read_text(encoding="utf-8") for name in MODULES}
    original=reference_constants(WORK/"base"/"generate_dataset.py")
    review_path=WORK/"augmentation_visual_review.json"
    reviews=json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    header='''"""Reproduce the original 1,000 cat maps and their targeted additions.

This standalone script embeds unmodified module sources with their own namespaces.
Commands base-pilot/base-full reproduce the original data. prepare-pilot,
select-pilot, approve-pilot, prepare-full, select-full and export add new maps.
validate checks the exported images, array, metadata, labels and original prefix.
Reference AI-review hashes apply only to exactly matching binary images.
No conditional flow matching model is trained and no train/test split is made.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import sys
import types
import linecache
from pathlib import Path
'''
    code=header+"\nMODULE_SOURCES = "+repr(sources)+"\n"
    code+='''
for _name,_source in MODULE_SOURCES.items():
    _filename=str(Path(__file__).resolve().parent/"_embedded"/f"{_name}.py")
    _module=types.ModuleType(_name)
    _module.__file__=_filename
    _module.__package__=""
    sys.modules[_name]=_module
    linecache.cache[_filename]=(len(_source),None,_source.splitlines(keepends=True),_filename)
    exec(compile(_source,_filename,"exec"),_module.__dict__)
'''
    for key,value in original.items():
        code+=f"sys.modules['release_workflow'].{key} = {value!r}\n"
    code+=f"sys.modules['augmentation_workflow'].REFERENCE_AUGMENTATION_REVIEW = {reviews!r}\n"
    code+='README_TEXT = '+repr((destination.parent/'README.md').read_text(encoding='utf-8'))+'\n'
    code+='''
FiniteSpecimenFEM=sys.modules["mechanics"].FiniteSpecimenFEM

if __name__=="__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    if len(sys.argv)>1 and sys.argv[1] in ("base-pilot","base-full"):
        sys.argv[1]=sys.argv[1].replace("base-","")
        sys.modules["release_workflow"].workflow_main()
    else:
        if len(sys.argv)>1 and sys.argv[1]=="validate" and "--output" not in sys.argv:
            sys.argv.extend(["--output",str(Path(__file__).resolve().parent)])
        _args=sys.modules["augmentation_workflow"].main()
        if _args.mode=="export":
            import shutil
            _source=Path(__file__).resolve()
            _target=(_args.output/"generate_dataset.py").resolve()
            if _source!=_target:
                shutil.copy2(_source,_target)
            (_args.output/"README.md").write_text(README_TEXT,encoding="utf-8")
'''
    compile(code,str(destination),"exec")
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(code,encoding="utf-8")
    write_readme(destination.parent)
    print(f"Standalone generator: {destination}, {len(code)} characters")


def write_readme(output):
    import csv
    metadata=output/'metadata.csv'
    n=len(list(csv.DictReader(metadata.open(encoding='utf-8')))) if metadata.exists() else 1250
    stage_counts={}
    for stage in ('pilot','full','styles'):
        path=WORK/f'{stage}_prepared.json'
        if path.exists():stage_counts[stage]=json.loads(path.read_text(encoding='utf-8')).get('source_count')
    original=(WORK/'base'/'README.md').read_text(encoding='utf-8')
    dependencies=original[original.index('参考环境：'):original.index('默认master seed=')]
    # Only retain the dependency installation block, not the old three-command workflow.
    dependencies=dependencies[:dependencies.index('python generate_dataset.py pilot')]+"```\n"
    sources=original[original.index('## Sources 与 licenses'):]
    text=f'''# Binary cat mechanical dataset — targeted augmentation

当前共 **{n} 张**224×224 binary material maps：原1000张加{n-1000}张独立新来源。`dataset.npy`形状为`({n},224,224)`、dtype为`uint8`，值只有0和1。**0=黑色硬固体，1=白色软固体；白色不是孔洞。** 没有训练conditional flow matching model，也没有划分training/validation/test集合。

## 文件与原数据保留

- `images/cat_0000.png`至`cat_{n-1:04d}.png`，按编号对应NumPy切片和CSV行。
- 前1000张的像素、文件名、seed和Ex/Ey标签保持原样；验证器检查原始像素及标签fingerprint。全局bin counts、sampling weights及preview成员随扩展更新。
- `metadata.csv`记录真实FEM标签、phase fraction、独立source seed、prompt strata、source/variant ID、目标模量与实际模量、检查结果。目标字段不是已达到的性能。
- `preview.png`是整个扩展集合随机100张的唯一10×10预览。
- `quality_report.md`报告实际覆盖、数值/视觉检查及限制；`generate_dataset.py`是包含生成、FEM、筛选和验证模块的独立脚本。

```python
import csv
import numpy as np
images = np.load('dataset.npy', allow_pickle=False)
with open('metadata.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
conditions = np.array([[float(r['Ex_over_Es']), float(r['Ey_over_Es'])]
                       for r in rows], dtype=np.float32)
```

PNG为mode`1`；Pillow读取为bool，转uint8后为0/1。一些8-bit解码器显示白色为255，仅是解码约定。CSV的64-bit seed应作为整数或字符串读取，避免电子表格截断。

## 固定力学协议

Esoft=1、Ehard=10，两相nu=.30；独立40×40×2mm方形试件，不周期阵列。两相各向同性、小应变linear elasticity、perfect bonding。Fully integrated Q4 plane stress，每个像素一个单元。

x加载：左边ux=0，右边ux=0.001×40mm；加载边uy自由，仅一个点约束刚体平移；上下边traction-free。y加载交换方向。`Ex_app=Fx/(Ly*t*nominal_strain)`，`Ey_app=Fy/(Lx*t*nominal_strain)`。它们是该协议下的apparent specimen moduli，不是periodic homogenization常数或C11/C22。

所有新候选的两个方向均实际FEM求解，使用double precision、AMG-preconditioned CG及1e-9代数容差；检查residual、energy/reaction consistency、force balance及fixed-fraction Reuss–Voigt必要界。固定相同nu和当前BC时，硬相比例f满足`1/(1-0.9f) <= Ex,Ey <= 1+9f`。必要界内部不意味着全部猫形可达。

## 定向补充方法

新master seed为1143186，与原始1142114不同。pilot/full使用独立随机流；额外style批次使用1144186。实际pilot sources={stage_counts.get('pilot',100)}，full sources={stage_counts.get('full','见质量报告/运行缓存')}，style sources={stage_counts.get('styles','见质量报告/运行缓存')}。除新的躯干、腿、尾巴、视角及outline/hollow contour/cut-paper外，根据用户风格参考加入binary illustration、pixel art及graphic cat portrait，当前三类各保留25张。Pixel art严格使用56或112网格，输出为常量4×4或2×2像素块；portrait允许头、胸部和前爪构图，使用独立的portrait semantic screening。风格类别有交叉，不代表互斥的人工标注。参考图用于明确风格，没有复制其像素。生成器仍为固定revision的本机Z-Image-Turbo，512px、9steps、batch4。

每个新source可产生最多3个uncropped isotropic extent/placement候选，逐一FEM计算；最终每个source最多保留1张。局部参数搜索用于选择性能，不能用来扩充独立样本数量。保留原1000张，优先补高模量和各向异性稀疏邻域，同时保留相似性能的不同几何。

新候选黑色比例允许5%–99%，实际猫bbox最长边至少56px，新style批次至少84px，并须通过自动semantic screening及最终实际像素AI视觉审查；没有人为填充noise/frame/pattern。部分高占比图可能是较小猫或outline猫，部分ink插画的躯干轮廓较稀疏，需结合预览判断。先完成50张定向pilot的技术/FEM/去重/视觉检查再进入full。50张pilot计入新增集合。

Near-duplicate规则与原数据一致：同背景polarity、黑色比例差<=.025、模量L-infinity距离<=.12，同时满足反射感知silhouette IoU>=.91，或IoU>=.85且pHash<=6。所有new-original和new-new配对检查；同seed variants互斥。实际模量接近但几何不同可保留。

请求的geometry/pose/view/style是prompt strata，不是经过人工确认的解剖标签。semantic score来自object-normalized source；入选图另按其实际二值像素AI复核。没有human visual inspection的声明。

## 依赖与复现

{dependencies}
脚本将模块源码嵌入独立namespace，不依赖项目目录的其他Python文件；Windows multiprocessing和FEM source fingerprint已检查。GPU、软件版本和batch size改变可能导致像素变化。

先从本次扩展数据提取仍保持不变的原1000张，作为独立`BASE`目录；无需依赖临时备份。令`WORK`为外部scratch。以下命令在脚本所在目录执行：

```text
python generate_dataset.py extract-base --base . --output BASE
python generate_dataset.py prepare-pilot --base BASE --work-dir WORK --sources 100
python generate_dataset.py select-pilot --base BASE --work-dir WORK
python generate_dataset.py approve-pilot --base BASE --work-dir WORK
python generate_dataset.py prepare-full --base BASE --work-dir WORK --sources {stage_counts.get('full',350)}
python generate_dataset.py prepare-styles --base BASE --work-dir WORK --sources {stage_counts.get('styles',120)} --seed 1144186
python generate_dataset.py select-full --base BASE --work-dir WORK --additions {n-1000}
python generate_dataset.py export --base BASE --work-dir WORK --output REPRODUCED --additions {n-1000}
python generate_dataset.py validate --output REPRODUCED
```

如需先重建原1000张，运行`base-pilot --work-dir BASE_WORK`和`base-full --work-dir BASE_WORK --output BASE`，均使用脚本内嵌的原版本流程及reference视觉审查。模型不调用外部收费生成API，checkpoint需预先下载或允许网络获取。

参考视觉审查按pixel SHA嵌入脚本。只有像素完全一致才能复用；其他seed/硬件产生新图时，需新的AI审查记录`WORK/augmentation_visual_review.json`，包含`passed_hashes`和`excluded_hashes`。未审查的pilot无法批准，未审查的full无法导出。代码不会把prompt或semantic score当成完整视觉审查。

`validate`检查所有PNG/NumPy/CSV一致、计数、二值、pixel hash、phase fraction、原1000标签及像素不变、FEM protocol/diagnostics、sampling weights和preview成员；它不重算全部FEM。独立复算可`from generate_dataset import FiniteSpecimenFEM`，在`threadpool_limits(1)`下调用`solver.solve(images[i])`。

之后训练时，`balanced_sampling_weight`需按真正training subset重新计算；限制稀疏区域过度重复采样，同source衍生图应归同一split。当前尚未执行任何split或模型训练。未来生成图在二值化后必须重新FEM验证，不能把输入condition当作实现性能。

{sources}
'''
    (output/'README.md').write_text(text,encoding='utf-8')


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    build(parser.parse_args().destination)
