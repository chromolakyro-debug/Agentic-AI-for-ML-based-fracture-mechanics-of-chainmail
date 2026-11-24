"""Create and independently verify a portable CFM v0 audit ZIP (standard library only)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = 'cfm_v0_audit/'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP members'
        assert all(n.startswith(PREFIX) and '..' not in PurePosixPath(n).parts and '\\' not in n for n in names)
        manifest_bytes = archive.read(PREFIX + 'AUDIT_MANIFEST.json')
        manifest = json.loads(manifest_bytes)
        expected = {PREFIX + row['path'] for row in manifest['files']}
        expected |= {PREFIX + 'AUDIT_MANIFEST.json', PREFIX + 'SHA256SUMS.txt'}
        assert set(names) == expected, 'Unexpected or missing ZIP members'
        checksums = dict(line.split('  ', 1)[::-1] for line in archive.read(PREFIX+'SHA256SUMS.txt').decode('utf-8').splitlines())
        assert checksums['AUDIT_MANIFEST.json'] == hashlib.sha256(manifest_bytes).hexdigest()
        assert set(checksums) == {row['path'] for row in manifest['files']} | {'AUDIT_MANIFEST.json'}
        total = 0
        for i, row in enumerate(manifest['files'], 1):
            digest, size = hashlib.sha256(), 0
            # Reading every member to EOF also validates its ZIP CRC.
            with archive.open(PREFIX + row['path']) as stream:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    digest.update(chunk)
                    size += len(chunk)
            assert size == row['bytes'], row['path']
            assert digest.hexdigest() == row['sha256'] == checksums[row['path']], row['path']
            total += size
            if i % 2000 == 0:
                print(f'Verified SHA256 + CRC: {i}/{len(manifest["files"])}', flush=True)
    return dict(payload_files=len(manifest['files']),zip_members=len(names),
                uncompressed_payload_bytes=total,all_payload_sha256_match=True,all_member_crc_pass=True,
                exact_member_inventory_pass=True)


def package(destination):
    if destination.exists() or destination.with_suffix(destination.suffix+'.partial').exists():
        raise FileExistsError('Choose a new output name; existing audit packages are preserved.')
    run = HERE/'run'
    audit = json.loads((run/'final_audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='complete' and audit['final_images']==audit['full_fem_valid']==640
    protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    frozen = json.loads((run/'frozen_protocol_sha256.json').read_text(encoding='utf-8'))
    final = json.loads((run/'final'/'manifest.json').read_text(encoding='utf-8'))
    recorded_code = json.loads((run/'source_sha256_final.json').read_text(encoding='utf-8'))
    for name, expected in recorded_code.items():
        assert digest_file(ROOT/name.replace('\\','/')) == expected, f'Changed experiment source: {name}'
    selected = list(HERE.glob('*.py')) + list(HERE.glob('*.md')) + list(HERE.glob('*.png'))
    selected += [HERE/'requirements.txt', HERE/'run_commands.ps1', HERE/'.gitignore']
    omitted_names = {'development_001000.pt','development_midpoint.pt','audit_package_validation.json'}
    selected += [p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in omitted_names]
    dataset = ROOT/'cat_binary_dataset'
    pngs = sorted((dataset/'images').glob('*.png'))
    assert [p.name for p in pngs] == [f'cat_{i:04d}.png' for i in range(8000)]
    selected += pngs + [dataset/n for n in ('dataset.npy','metadata.csv','README.md','quality_report.md','preview.png','generate_dataset.py')]
    selected += [ROOT/n for n in ('mechanics.py','evaluate_mechanics.py','mechanics_checks.py',
                                  'mechanics_selection.py','reference_style_selection.py','augmentation_selection.py')]
    selected = sorted(set(selected), key=lambda p: p.relative_to(ROOT).as_posix())
    assert all(p.is_file() and not p.is_symlink() and ROOT in p.resolve().parents for p in selected)
    rows = []
    for i, path in enumerate(selected, 1):
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest_file(path)))
        if i % 2000 == 0:print(f'Inventoried {i}/{len(selected)}',flush=True)
    by_name = {r['path']:r for r in rows}
    for name, expected in {
        'cat_binary_dataset/dataset.npy':protocol['data_sha256'],
        'cat_binary_dataset/metadata.csv':protocol['metadata_sha256'],
        'cfm_v0/run/protocol.json':frozen['sha256'],
        'cfm_v0/run/checkpoints/best.pt':final['checkpoint_sha256'],
    }.items():assert by_name[name]['sha256']==expected, f'Frozen evidence mismatch: {name}'
    readme = '''# CFM v0 审计资料包

这是已完成的 CFM v0 实验快照。先阅读 `cfm_v0/REPORT.md`；三张核心图就在同一目录。

本次实测：RTX 4090，20,000 training steps，33.42 分钟；20 个固定目标各 32 个 noise seeds，共 640 张最终 binary maps 全部完成原 FEM。联合属性命中 244/640（38.1%），training-only retrieval 命中 20/20 个目标。固定 40 张的 AI 实际视觉检查中，20 张基本连贯、19 张失真、1 张难以辨猫；5 张同时满足基本连贯与属性命中。限制和解释以报告为准。

## 资料导航

- `cat_binary_dataset/`：原始 8,000 张 PNG、完整 dataset.npy、metadata.csv、说明、质量报告、preview 和原数据生成脚本。像素、行序、FEM 标签保持原样。
- `cfm_v0/*.py`、`requirements.txt`、`run_commands.ps1`：CFM 实现、数据审计、训练/恢复、采样、FEM 调用、统计、图表和复现入口。
- 根目录 mechanics/selection 模块：本轮实际复用的 FEM solver、独立数值检查及近邻描述子实现，保留原目录关系。
- `cfm_v0/run/protocol.json`、`splits.json`、`data.npz`：冻结的目标、support scores、noise seeds、阈值、数据划分及 training-only 统计。
- `cfm_v0/run/checkpoints/best.pt`：实际最终采样的 EMA checkpoint；`last.pt`：含 optimizer、RNG 等状态的可恢复训练 checkpoint。
- `cfm_v0/run/final/`：640 张 binary arrays、逐目标缓存、逐样本 CSV/JSON、每目标汇总、全部 FEM diagnostics、cross-coupling、retrieval/近邻结果、阈值边界诊断和绑定 pixel hash 的视觉审查记录。
- `cfm_v0/run/development_*`、`smoke_*`、`training.jsonl`、`environment.json` 等：development/smoke 检查、训练日志、软件版本、恢复检查和完成审计。

## 校验

`AUDIT_MANIFEST.json` 记录每个资料文件的相对路径、字节数和 SHA256。`SHA256SUMS.txt` 同时覆盖资料和 manifest；它不对自身求 hash。ZIP 旁的 `.sha256` 文件校验整个 ZIP。

解压后，可用任意 Python 3 运行标准库校验，不需要 GPU：

```text
python cfm_v0/package_audit.py --verify <本ZIP的路径>
```

该命令逐文件读取 ZIP，核对 SHA256、CRC、长度和精确成员清单。已有实验结果的一致性复查入口为 `python cfm_v0/audit_final.py`（需要 NumPy/Pillow）。

## 复现边界

未打包 `.venv`、`__pycache__`、两个中间 development checkpoint、与本轮无关的历史工程输出，以及原数据生成用的外部 diffusion/SigLIP 权重。训练输入本身已经完整包含，CFM 复现无需重新生成原始数据。环境按 `requirements.txt` 重建；不同软硬件不保证 bitwise 一致。

复现时先设置 `CFM_V0_RUN_DIR` 为新目录，再运行 `run_commands.ps1`，保留本次冻结结果。历史 command logs 中的绝对路径是原运行记录；实际脚本根据解压位置定位工程。`record_visual_review.py` 仅绑定本次实际查看过的 manifest，不能将旧判断自动套用到新生成结果。
'''.encode('utf-8')
    rows.append(dict(path='AUDIT_README.md',bytes=len(readme),sha256=hashlib.sha256(readme).hexdigest()))
    manifest = dict(format_version=1,project='Cat shaped mechanical metamaterials',experiment='CFM v0',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    contents='Frozen input dataset, experiment sources, final/resume checkpoints, all run evidence, report and figures',
                    omitted=['.venv','__pycache__','two intermediate development checkpoints','unrelated historical outputs',
                             'external dataset-generation model weights','external archive validation sidecar'],
                    original_source_fingerprints_verified=True,original_dataset_and_protocol_fingerprints_verified=True,
                    files=rows)
    manifest_bytes = encoded(manifest)
    sums = ''.join(f'{r["sha256"]}  {r["path"]}\n' for r in rows)
    sums += hashlib.sha256(manifest_bytes).hexdigest()+'  AUDIT_MANIFEST.json\n'
    temporary = destination.with_suffix(destination.suffix+'.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for i,path in enumerate(selected,1):
            archive.write(path,PREFIX+path.relative_to(ROOT).as_posix())
            if i%2000==0:print(f'Compressed {i}/{len(selected)}',flush=True)
        archive.writestr(PREFIX+'AUDIT_README.md',readme)
        archive.writestr(PREFIX+'AUDIT_MANIFEST.json',manifest_bytes)
        archive.writestr(PREFIX+'SHA256SUMS.txt',sums.encode('utf-8'))
    validation = verify(temporary)
    temporary.replace(destination)
    digest = digest_file(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n',encoding='utf-8')
    validation.update(archive=str(destination),zip_bytes=destination.stat().st_size,zip_sha256=digest,
                      created_utc=manifest['created_utc'],original_frozen_inputs_verified=True)
    (run/'audit_package_validation.json').write_bytes(encoded(validation))
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'cfm_v0_audit_2026-11-14.zip')
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify(args.verify),indent=2))
    else:package(args.output.resolve())
