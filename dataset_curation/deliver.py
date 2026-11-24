"""Write the single requested comparison figure, concise report and usage notes."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from curate import ROOT,OUT,HERE,read,save,read_csv,sha


def main():
    checks=read(OUT/'validation.json');loader=read(OUT/'loader_validation.json')
    assert checks['status']==loader['status']=='passed'
    classes=read(OUT/'classes.json')['classes'];_,rows=read_csv(OUT/'metadata.csv')
    rng=np.random.default_rng(20261116)
    canvas=Image.new('RGB',(1908,1490),'#f3f5f7');draw=ImageDraw.Draw(canvas)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    draw.text((20,12),'猫形机械超材料：八类整理结果 / 7,857 张',font=font(30),fill='#173345')
    draw.text((20,56),'每类均匀随机抽取4张，seed=20261116；只按原requested_style映射分组，不按单图外观筛选。',font=font(19),fill='#50616b')
    chosen=[]
    for j,c in enumerate(classes):
        x=18+(j%2)*948;y=102+(j//2)*344
        draw.text((x,y),f'{c["group_id"]}  {c["group_name_zh"]}  ·  {c["count"]:,} 张  |  {c["group_name_en"]}',font=font(22),fill='#173345')
        members=[i for i,r in enumerate(rows) if r['group_id']==c['group_id']]
        picks=rng.choice(members,4,replace=False).tolist()
        for k,i in enumerate(picks):
            r=rows[i];xx=x+k*234;yy=y+39
            with Image.open(OUT/r['new_relative_path']) as im:
                assert im.size==(224,224);canvas.paste(im.convert('RGB'),(xx,yy))
            draw.rectangle((xx-1,yy-1,xx+224,yy+224),outline='#ccd2d7')
            draw.text((xx,yy+228),r['new_id'],font=font(16),fill='#173345')
            draw.text((xx,yy+251),'原 '+r['original_filename'],font=font(14),fill='#50616b')
            chosen.append(dict(dataset_index=i,new_id=r['new_id'],path=r['new_relative_path'],
                original_id=r['original_id'],requested_style=r['requested_style'],sha256_pixels=r['sha256_pixels']))
    draw.text((20,1460),'0=黑色硬材料；1=白色软材料。对照图仅拼排原像素；有效图片文件均为原PNG字节副本。',font=font(16),fill='#50616b')
    canvas.save(OUT/'eight_classes.png')
    save(OUT/'provenance'/'figure_samples.json',dict(seed=20261116,selection='Uniform random4 without replacement within each class; no visual/FEM filter',samples=chosen))
    splits=read(OUT/'splits.json');summary=read(OUT/'provenance'/'curation_summary.json');migration=read(OUT/'provenance'/'split_migration.json')
    report=['# 数据集整理报告','',
      '完整原始8,000张、64个requested_style细类已逐一核验。按用户指定的精确标签映射，51个细类合并为八组、保留7,857张；13个细类整体排除143张。没有未知标签、数量差异、单图例外或视觉筛选。原数据和既有实验结果/checkpoint保留。', '',
      '| 编号 | 新类别 | 数量 | Train | Validation | Test |','|---|---|---:|---:|---:|---:|']
    for c in classes:
        n=splits['class_counts'][c['group_id']]
        report.append(f'| {c["group_id"]} | {c["group_name_zh"]} / {c["group_label"]} | {c["count"]:,} | {n["train"]:,} | {n["validation"]} | {n["test"]} |')
    report+=['| 合计 | | 7,857 | 6,285 | 786 | 786 |','',
      '图片位于`images/01_realistic_stippling/`至`images/08_silhouette/`。每类按原始数值ID升序，从`cat_000001.png`连续编号；全局唯一`new_id`使用`01_cat_000001`形式。NPY、NPZ、CSV、JSONL和split已按同一新索引同步重排。', '',
      '所有保留图片逐文件SHA256与原图一致，原PNG格式、尺寸、像素和黑白相位未改。按原始ID与pixel hash建立关联，再逐样本核对原CSV和冻结CFM数据的Ex/Ey及s/a；所有力学数值原样复制，没有重新FEM。', '',
      '原metadata全部124个字段与原值保存在`metadata.jsonl`的`original_metadata`中，原CSV还以原字节完整存入`provenance/original_dataset/metadata.csv`。有效`metadata.csv`只改写原`filename`为新images目录下路径，其余原字段不变，另增加新旧ID、路径、类别与split追溯字段。完整删除记录在`provenance/deleted_samples.*`；8,000条操作映射在`provenance/all_sample_actions.csv`。', '',
      f'根据用户对冲突条款的明确答复，采用按八类重新分层80/10/10，seed=20261116；整数取整后为6,285/786/786。三个新集合互斥且覆盖全部保留样本。相对旧split，{migration["migrated"]:,}张保留样本发生集合迁移；因此不宣称满足已被替代的“只过滤、不迁移”条款。旧split原件及其过滤后版本均保留在provenance，且不作为有效split。', '',
      '原CFM mean/std、物理定义和数值保持不变。mean/std来自原6400张training set，不是按新split拟合；旧checkpoint也已见过部分新validation/test，因此新split不能直接当作旧模型的全新holdout。旧sampling weights与support统计完整归档，并按原ID保存保留样本的历史值，没有把缺失旧权重补造给新train，也未重新计算采样权重或support。', '',
      '验收通过：逐图字节/像素一致、原metadata无损、标签对应正确、新ID与路径有效、删除类不在有效索引/split中、原文件保留。实际调用现有`cfm_v0.train.setup(data_dir=..., run_dir=...)`读取全部7,857行，检查八类和split，并构造16张CPU输入batch；默认读取原8,000张仍通过。唯一既有代码改动是为该loader增加可选路径参数，默认行为不变；修改前代码已归档。', '',
      f'实测源核验约{summary["source_audit_elapsed_seconds"]:.1f}秒，整理约{summary["elapsed_seconds"]:.1f}秒，完整读回验收约{checks["elapsed_seconds"]:.1f}秒。未重新生成图片、未重新计算力学性质、未训练模型。八类对照图见`eight_classes.png`，每类4张固定随机样本；完整检查记录为`validation.json`与`loader_validation.json`。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('''# Cat-shaped Mechanical Metamaterials — curated v1

这是完整原始数据按指定requested_style整类整理后的新版本：7,857张、八类。原8,000张数据和CFM实验仍在项目原目录中。类别与数量见classes.json，整理结果见REPORT.md。

目录与主键：

```text
images/
  01_realistic_stippling/cat_000001.png ... cat_000750.png
  02_realistic_ink_drawing/cat_000001.png ... cat_001500.png
  03_triangular_geometry/cat_000001.png ... cat_000750.png
  04_fragmented_geometry/cat_000001.png ... cat_000750.png
  05_pixel_line_art/cat_000001.png ... cat_000759.png
  06_solid_pixel_art/cat_000001.png ... cat_000766.png
  07_cartoon/cat_000001.png ... cat_001500.png
  08_silhouette/cat_000001.png ... cat_001082.png
dataset.npy        # (7857,224,224) uint8, 0 black hard / 1 white soft
metadata.csv       # 全部原字段 + 新字段；filename已改为images/下的新路径
metadata.jsonl     # curation新关联 + original_metadata完整原始字段和值
index.csv          # 有效样本主键、全局数组索引、新路径与split
data.npz           # 原值Ex/Ey、s/a、mean/std、八类label、新split与显式样本ID/路径
splits.json        # 当前有效split（新分层），索引从0开始
classes.json       # group_id 01..08，class_labels 0..7
normalization.json # 冻结原CFM参数及适用范围
provenance/        # 原metadata/split/data.npz、完整删除及操作记录；不是有效数据索引
```

`new_filename`在类内唯一，`new_id`（如01_cat_000001）全局唯一。类内数字从1开始，数组索引从0开始。数组总体顺序为group_id升序，再按原始数值ID升序。不能只用不同目录下相同的cat_000001文件名作全局主键。

两种有效图片路径写法等价：`root / row['new_relative_path']`或`root / 'images' / row['filename']`。`original_*`字段明确指向原始数据身份，保留作追溯，不是新图片加载路径。`reference_filename`仍是原风格参考身份；对应原字节副本位于`curation.reference_relative_path`。

全部原CSV值保留字符串形式，包括64-bit seed、空字符串和JSON字段。`metadata.csv`的原filename是唯一被改写的原字段，完整旧值在`original_filename`和JSONL的`original_metadata.filename`中。CSV的`validation`列仍表示原数据质量检查pass；实际集合归属是`curation.split`。旧`preview_cell`、balanced_sampling_weight和生成阶段字段等也保持历史值，不代表本轮新预览或新train统计。

最小读取示例（不训练）：

```python
from pathlib import Path
import csv
import numpy as np
from PIL import Image

root = Path('cat_binary_dataset_curated_v1')
images = np.load(root / 'dataset.npy', mmap_mode='r', allow_pickle=False)
data = np.load(root / 'data.npz', allow_pickle=False)
with (root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
i = int(data['train'][0])
material_map = images[i]
same_image = Image.open(root / rows[i]['new_relative_path'])
Ex, Ey = data['moduli'][i]
s, a = data['sa'][i]
class_label = int(data['class_labels'][i])  # 0..7
new_id = str(data['sample_ids'][i])
```

项目现有CFM reader已增加可选路径参数，默认仍读取原数据：

```python
import sys
from pathlib import Path
project = Path.cwd()  # 从项目根目录运行
sys.path.insert(0, str(project / 'cfm_v0'))
from train import setup
root = project / 'cat_binary_dataset_curated_v1'
data, images = setup(data_dir=root, run_dir=root)
```

本目录是数据发布目录，不是可原地覆盖的训练run。旧train_weights、kNN density和旧support不能直接套给新split，因此这些内容只在provenance保留；当前data.npz不提供伪造或错位的train_weights。新训练所需的sampler/support应在单独实验中明确配置。本轮按用户要求保留旧mean/std；旧参数和旧checkpoint不应被宣称从未接触新validation/test。

完整核验命令（从项目根目录，在现有环境运行）：

```powershell
.\\cfm_v0\\.venv\\Scripts\\python.exe dataset_curation\\validate.py
.\\cfm_v0\\.venv\\Scripts\\python.exe dataset_curation\\verify_existing_loader.py
```

生成脚本为dataset_curation/curate.py，audit先核对全源，build执行精确映射。已完成的数据集禁止原地重新build。classification_rules.json保留完整64标签的显式映射。没有基于图片外观改类、重复补足数量、FEM重新求解或模型训练。
''',encoding='utf-8')
    # Include the small curation implementation for independent inspection.
    code=OUT/'provenance'/'curation_code';code.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):
        import shutil
        shutil.copy2(p,code/p.name)
    save(OUT/'provenance'/'execution_commands.json',dict(
        cwd=str(ROOT),python=str(ROOT/'cfm_v0'/'.venv'/'Scripts'/'python.exe'),
        commands=['dataset_curation/curate.py audit','dataset_curation/curate.py build',
                  'dataset_curation/validate.py','dataset_curation/verify_existing_loader.py','dataset_curation/deliver.py'],
        outcome='All listed commands actually executed; read-back and loader reports passed; no FEM or model training command invoked.'))
    print(OUT/'REPORT.md');print(OUT/'eight_classes.png')


if __name__=='__main__':main()
