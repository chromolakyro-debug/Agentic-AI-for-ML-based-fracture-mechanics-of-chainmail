"""Package reference-style expansion and previous reproducible stages together."""
import tempfile
from pathlib import Path
import argparse
import ast
from collections import Counter
import csv
import json

ROOT=Path(__file__).resolve().parent
WORK=(Path(tempfile.gettempdir()) / 'binary-cat-reference-expansion')


def read_literal_bundle(path):
    sources=None;setup=[]
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(node,ast.Assign):continue
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id=='MODULE_SOURCES':sources=ast.literal_eval(node.value)
            if isinstance(target,ast.Attribute) and target.attr.startswith('REFERENCE_'):
                parent=target.value
                if isinstance(parent,ast.Subscript):
                    try:setup.append((ast.literal_eval(parent.slice),target.attr,ast.literal_eval(node.value)))
                    except (ValueError,TypeError):pass
    if sources is None:raise ValueError('Original standalone sources not found')
    return sources,setup


def readme(output,work):
    original=(work/'base1250'/'README.md').read_text(encoding='utf-8')
    dependencies=original[original.index('参考环境：'):original.index('脚本将模块源码嵌入')]
    sources=original[original.index('## Sources 与 licenses'):]
    n=8000
    if (output/'metadata.csv').exists():
        with (output/'metadata.csv').open(encoding='utf-8',newline='') as f:n=len(list(csv.DictReader(f)))
    stage={}
    for mode in ('pilot','full'):
        p=work/f'{mode}_generation_state.json'
        if p.exists():stage[mode]=json.loads(p.read_text(encoding='utf-8'))['source_count']
    text=f'''# Binary cat mechanical dataset — reference-style expansion

当前共**{n}张**，原1250张的像素、文件名、seed、Ex/Ey保持不变；新增{n-1250}张只来自本轮用户提供的9类风格。`dataset.npy`为`({n},224,224)`、`uint8`、值只有0/1。**0=黑色硬固体，1=白色软固体；白色不是孔洞。** 当前没有训练模型，也没有划分training/test数据。

## 输出与使用

- `images/cat_0000.png`至`cat_{n-1:04d}.png`；NumPy切片与文件编号、CSV行顺序一致。
- `metadata.csv`含实际FEM标签、黑色面积比例、source seed、prompt、reference文件名/hash、style strata、QC和数值诊断。
- `preview.png`是全数据集随机100张、10×10的唯一最终预览；`quality_report.md`给出实测覆盖和局限。
- `generate_dataset.py`独立封装本轮和旧版本生成/FEM/选择代码。PNG为one-bit；Pillow读入bool转uint8为0/1，8-bit查看器可能显示为0/255。

```python
import csv
import numpy as np
images=np.load('dataset.npy',allow_pickle=False)
with open('metadata.csv',encoding='utf-8',newline='') as f:
    rows=list(csv.DictReader(f))
conditions=np.array([[float(r['Ex_app']),float(r['Ey_app'])] for r in rows],dtype=np.float32)
new_maps=images[1250:]
```

CSV中的64-bit seed应读作整数或字符串，不能经过浮点电子表格截断。`balanced_sampling_weight`只是完整数据集的建议权重；未来真正训练时应按training subset重新计算。

## 风格与生成方法

新增样本参考项目`cat-style-references/`中的02三角网格、03钢笔插画、04outline pixel art、06whiskered sprite、07木刻、08漫画、09几何拼块、10halftone、11反相线条。05与用户要求停止增加的旧剪影风格重合，故新增quota排除05。11的月亮、星星和外框不进入材料图。参考文件仅作为视觉风格来源，生成代码不复制其像素；实际文件hash写入metadata。

每种风格包含不同猫体比例、姿势/视角、whole-body或portrait构图、黑白背景与主体尺寸。每个独立source只保留一个输出，未靠重复旋转/平移/缩放的版本扩充数量。Pixel art保留实际常量像素块；halftone为猫自身的局部图像表达，并不在空背景加随机填充。所有最终像素严格binary。

使用与旧数据相同、固定revision的本机Z-Image-Turbo，512px、9steps，master seed20261114；pilot/full独立随机流，batch size4。先产生并检查50张pilot，再进行完整扩展。实际source counts：pilot={stage.get('pilot',144)}，full={stage.get('full','见生成命令/报告')}。每张候选实际FEM求解后，按style quota、性能覆盖、局部几何新颖度和重复冲突选择。没有训练surrogate或inverse模型，也没有把prompt中的目标当作实现模量。

自动semantic/style scores只是screening；最终新增实际像素经过AI contact-sheet review，对疑点放大检查。审查记录按pixel SHA绑定，不能复用到不同像素。没有human visual inspection或完美可识别性声明。Metadata中的pose/view/style是生成意图，不是经人工逐项确认的解剖标签。

## 固定FEM协议

Esoft=1、Ehard=10，两相nu=.30；独立40×40×2mm试件，不周期阵列。小应变、两相各向同性linear elasticity、perfect bonding、Q4 plane stress，每像素一个单元。x加载左右边施加0/0.001×40mm的ux，横向自由，仅单点消除刚体平移；上下边traction-free。y加载交换方向。Ex=Fx/(Ly*t*strain)，Ey=Fy/(Lx*t*strain)，是当前BC下的apparent specimen moduli。

Double precision、AMG-CG、1e-9代数容差；记录residual、energy/reaction、force-balance及fixed-fraction Reuss–Voigt必要界检查。若硬相比例f，则`1/(1-0.9f)<=Ex,Ey<=1+9f`；必要界内部不意味着所有猫形均可达。细纹理和像素角点的可制造性、3D/非线性/失效尚未验证。

## 依赖与复现

{dependencies}

脚本自含module source并保留独立namespace；不依赖项目目录中的其他Python文件。模型checkpoint已固定版本；GPU/软件/batch变化仍可能影响像素，不能承诺跨平台bitwise相同。

下面从当前数据提取原1250张，再复现新增部分。BASE、WORK、REPRODUCED应为不同目录，WORK在最终交付目录以外。

```text
python generate_dataset.py extract-base --base . --output BASE
python generate_dataset.py generate-pilot --base BASE --work-dir WORK --sources {stage.get('pilot',144)}
python generate_dataset.py label-pilot --base BASE --work-dir WORK
python generate_dataset.py select-pilot --base BASE --work-dir WORK
python generate_dataset.py approve-pilot --base BASE --work-dir WORK
python generate_dataset.py generate-full --base BASE --work-dir WORK --sources {stage.get('full',10080)}
python generate_dataset.py label-full --base BASE --work-dir WORK
python generate_dataset.py select-full --base BASE --work-dir WORK --total {n}
python generate_dataset.py export --base BASE --work-dir WORK --output REPRODUCED --total {n}
python generate_dataset.py validate --output REPRODUCED --total {n}
```

参考审查hash已嵌入脚本，完全相同的像素可复用；新seed/不同像素需要重新审查。`review-pilot`/`review-full --start A --end B --exclude "POSITION:reason"`只记录已经实际完成的AI审查，命令本身不会做视觉评估。未审查的pilot不能批准，未审查的full不能导出。`stream-full`可在GPU生成时用CPU并行标注，但同一个labels文件只能有一个writer；普通复现按上面顺序执行即可。

旧1250的完整流程仍被封装，可通过`legacy ...`访问原版本命令，例如`legacy base-pilot`和`legacy prepare-pilot`；本轮扩展不要求重新生成原图，直接提取原1250即可。

`validate`逐图检查尺寸、值域、PNG/NumPy/metadata、pixel hash、seed、物理参数、solver protocol、数值诊断、原1250身份、style标签、sampling weights和preview内容；它不重新求解全部FEM。独立复算可导入`FiniteSpecimenFEM`并在`threadpool_limits(1)`下调用`solve`。未来模型生成的二值图必须重新FEM验证。

{sources}

本轮另外使用用户提供的`cat-style-references`图作为风格参考，文件身份由metadata中的SHA256记录；未另外收集网络猫图。风格参考并非对其版权/授权范围的独立核验。
'''
    return text


def build(destination,work):
    sources,setup=read_literal_bundle(work/'base1250'/'generate_dataset.py')
    for name in ('reference_style_candidates','reference_style_selection','reference_style_workflow'):
        sources[name]=(ROOT/f'{name}.py').read_text(encoding='utf-8')
    reviews=json.loads((work/'reference_visual_review.json').read_text(encoding='utf-8')) if (work/'reference_visual_review.json').exists() else {}
    documentation=readme(destination.parent,work)
    code='''"""Self-contained reference-style binary cat dataset pipeline."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import sys,types,linecache
from pathlib import Path
'''
    code+='MODULE_SOURCES = '+repr(sources)+'\n'
    code+='''
for _name,_source in MODULE_SOURCES.items():
    _filename=str(Path(__file__).resolve().parent/'_embedded'/f'{_name}.py')
    _module=types.ModuleType(_name);_module.__file__=_filename;_module.__package__=''
    sys.modules[_name]=_module
    linecache.cache[_filename]=(len(_source),None,_source.splitlines(keepends=True),_filename)
    exec(compile(_source,_filename,'exec'),_module.__dict__)
'''
    for module,attr,value in setup:code+=f"sys.modules[{module!r}].{attr} = {value!r}\n"
    code+=f"sys.modules['reference_style_workflow'].REFERENCE_STYLE_REVIEW = {reviews!r}\n"
    code+='README_TEXT = '+repr(documentation)+'\n'
    code+='LEGACY_GENERATOR_TEXT = '+repr((work/'base1250'/'generate_dataset.py').read_text(encoding='utf-8'))+'\n'
    code+='LEGACY_README_TEXT = '+repr((work/'base1250'/'README.md').read_text(encoding='utf-8'))+'\n'
    code+='''
FiniteSpecimenFEM=sys.modules['mechanics'].FiniteSpecimenFEM
if __name__=='__main__':
    import multiprocessing,shutil
    multiprocessing.freeze_support()
    if len(sys.argv)>1 and sys.argv[1]=='legacy':
        del sys.argv[1]
        if len(sys.argv)>1 and sys.argv[1] in ('base-pilot','base-full'):
            sys.argv[1]=sys.argv[1].replace('base-','')
            sys.modules['release_workflow'].workflow_main()
        else:
            _legacy_args=sys.modules['augmentation_workflow'].main()
            if _legacy_args.mode=='export':
                (_legacy_args.output/'generate_dataset.py').write_text(LEGACY_GENERATOR_TEXT,encoding='utf-8')
                (_legacy_args.output/'README.md').write_text(LEGACY_README_TEXT,encoding='utf-8')
    else:
        if len(sys.argv)>1 and sys.argv[1]=='validate' and '--output' not in sys.argv:
            sys.argv.extend(['--output',str(Path(__file__).resolve().parent)])
        _args=sys.modules['reference_style_workflow'].main()
        if _args.mode=='export':
            _source=Path(__file__).resolve();_target=(_args.output/'generate_dataset.py').resolve()
            if _source!=_target:shutil.copy2(_source,_target)
            (_args.output/'README.md').write_text(README_TEXT,encoding='utf-8')
'''
    compile(code,str(destination),'exec')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(code,encoding='utf-8')
    (destination.parent/'README.md').write_text(documentation,encoding='utf-8')
    print(f'Standalone generator built: {destination}; {len(code)} characters')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path)
    p.add_argument('--work-dir',type=Path,default=WORK);args=p.parse_args();build(args.destination,args.work_dir)
