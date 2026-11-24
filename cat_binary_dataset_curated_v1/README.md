# Cat-shaped Mechanical Metamaterials — curated v1

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
.\cfm_v0\.venv\Scripts\python.exe dataset_curation\validate.py
.\cfm_v0\.venv\Scripts\python.exe dataset_curation\verify_existing_loader.py
```

生成脚本为dataset_curation/curate.py，audit先核对全源，build执行精确映射。已完成的数据集禁止原地重新build。classification_rules.json保留完整64标签的显式映射。没有基于图片外观改类、重复补足数量、FEM重新求解或模型训练。
