"""Small self-contained review package; omit weights, full datasets, raw floats and caches."""
import json,zipfile,datetime
from pathlib import Path
import numpy as np
from utils import HERE,ROOT,read,save,sha,csvwrite,adapter

def main():
    cfg=read(HERE/'configs/old_20k.json');ds=adapter(cfg['dataset']['adapter'])(cfg['dataset'],ROOT)
    csvwrite(HERE/'comparison/training_catalog.csv',[ds.describe(int(i)) for i in ds.train]);save(HERE/'comparison/training_splits.json',ds.splits)
    for tag in ['old_20k','old_125k']:
        out=HERE/'runs'/tag;rows=read(out/'results.json');arrays={};binary=[]
        for r in rows:
            if r['raw_array'] not in arrays:arrays[r['raw_array']]=np.load(out/r['raw_array'],allow_pickle=False)['binary']
            binary.append(arrays[r['raw_array']][r['array_index']])
        np.savez_compressed(out/'fem_inputs.npz',binary=np.stack(binary),sample_keys=np.array([r['sample_key'] for r in rows]),seeds=np.array([r['seed'] for r in rows]))
    start='''# 从这里开始

本包审计猫形机械超材料CFM evaluation pipeline，不含训练任务。两个既有EMA checkpoint：20k与此前最终选定的125k；旧8,000张数据、旧6,400张训练集。用户红圈24格来自整理后八类数据的原6×6等宽s–a网格，坐标使用精确bin edges计算。

先读REPORT.md，再分别打开runs/old_20k/index.html和runs/old_125k/index.html。每个页面24个面板，每面板固定10张生成、10张target retrieval、10张best-generated-FEM retrieval。显示旋转镜像不改变原始FEM值。第三行是经过FEM筛选的best-of-10，不能代替普通生成质量。

重点审阅：固定全局轴下的机械误差与交叉耦合；相同seed跨条件的形状变化；猫形可辨识性、完整性和多样性；train-only轴交换检索是否公平；有限试件FEM与条件定义是否一致。视觉和memorization没有自动评分，留给人工，机械近邻不能证明或否定记忆。

configs/保存targets、bin edges、用户圈选图、seeds、模型/采样/FEM设置。comparison/含6400张训练样本属性目录、旧split、两模型逐target对照和描述性scan slopes；该目录足以在不读取完整训练图片的情况下复算两组retrieval。runs/包含逐样本FEM结果、solver diagnostics、summary、HTML、图片、精确二值fem_inputs.npz、manifest和运行代码快照。validation/包含核心测试、真实FEM轴交换检查、缓存/重采样验证及浏览器交互检查。

所有480张正式生成的连续raw float32输出、10份初始noise及逐target NPZ完整保存在本地evaluation/runs/，为控制包大小不包含这些大数组。模型权重和全量原图未打包，路径与SHA256见manifest。没有删掉失败样本；结果表仍是全部固定seeds。

解压后运行`python verify_ai_archive.py`（需要NumPy和Pillow），可独立核对全部文件hash、二值图/FEM输入对应、误差、ddof=1方差、train-only去重top10及HTML数值。此脚本核对保存的FEM结果，不声称重新执行FEM；重算求解需原项目环境或利用包内mechanics.py代码快照恢复环境。
'''
    (HERE/'START_HERE.md').write_text(start,encoding='utf-8')
    files=[]
    for p in HERE.glob('*.py'):files.append(p)
    for name in ['START_HERE.md','REPORT.md','README.md']:files.append(HERE/name)
    for folder in ['configs','adapters','comparison']:
        files += [p for p in (HERE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    # Include small evidence and logs; omit browser test copies/screenshots/profiles.
    files += [p for p in (HERE/'validation').glob('*') if p.is_file() and p.suffix in ['.json','.log']]
    for tag in ['old_20k','old_125k']:
        out=HERE/'runs'/tag
        files += [p for p in out.rglob('*') if p.is_file() and 'arrays' not in p.relative_to(out).parts and p.name!='initial_noise.npy']
    files=sorted(set(files));inventory=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=[dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    save(HERE/'ARCHIVE_INVENTORY.json',inventory);files.append(HERE/'ARCHIVE_INVENTORY.json')
    destination=HERE/'evaluation_ai_review.zip';temp=destination.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(HERE).as_posix())
    temp.replace(destination);save(HERE/'audit_package.json',dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),files=len(files)))
    print(f'{destination}: {destination.stat().st_size/1024**2:.2f} MiB, {len(files)} files')

if __name__=='__main__':main()
