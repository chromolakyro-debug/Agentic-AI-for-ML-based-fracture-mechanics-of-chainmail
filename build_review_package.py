"""Build a compact Chinese review ZIP only from an already validated publication.

No dataset, analysis, model, review manifest or candidate cache is modified.
The scratch staging directory is retained; only the requested ZIP is published.
Requires NumPy and Pillow. This is a review package, not the complete dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_WORK = (Path(tempfile.gettempdir()) / 'binary-cat-reference-expansion')
COUNT = 8000
BASE_COUNT = 1250
DELIVERABLES = ('dataset.npy', 'metadata.csv', 'preview.png', 'quality_report.md',
                'generate_dataset.py', 'README.md')
FAMILIES = (
    ('02', 'triangular_lattice', '三角 lattice'),
    ('03', 'binary_pen', 'Binary pen illustration'),
    ('04', 'outline_pixel', 'Outline pixel art'),
    ('06', 'whiskered_sprite', 'Whiskered pixel sprite'),
    ('07', 'woodcut', 'Woodcut / engraving'),
    ('08', 'comic', 'Comic cartoon'),
    ('09', 'geometric', 'Geometric cat'),
    ('10', 'halftone', 'Halftone / clustered dots'),
    ('11', 'inverted_line', '反相 tabby stencil'),
)
EVIDENCE = (
    'publication_state.json', 'finalized_staging.json', 'package_export_check.json',
    'base_identity.json', 'release_identity.json', 'base_pair_audit.json',
    'final_pair_audit.json', 'pilot_generation_state.json', 'full_generation_state.json',
    'pilot_refinement.json', 'refinement.json', 'refinement_plan.json',
    'reproduction.json', 'reproduction_details.json', 'reference_visual_review.json',
    'final_visual_correction.json',
)
ANALYSIS_FILES = (
    'reference_expansion_statistics.json', 'reference_expansion_comparison.png',
    'reference_expansion_comparison.pdf', 'style_examples.png', 'style_examples_index.json',
)
ANALYSIS_SCRIPTS = ('compare_reference_expansion.py', 'compare_augmentation.py',
                    'show_reference_styles.py')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pixel_hash(pixels):
    return hashlib.sha256(np.asarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def safe_member(name):
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and '\\' not in name
            and ':' not in name and all(p not in ('', '.', '..') for p in name.split('/')),
            f'Unsafe archive member: {name!r}')
    return name


def png_pixels(path):
    with Image.open(path) as image:
        require(image.size == (224, 224), f'Wrong PNG dimensions: {path.name}')
        values = np.asarray(image.convert('L'))
    require(set(np.unique(values)) == {0, 255}, f'Not binary black/white: {path.name}')
    return (values == 255).astype(np.uint8)


def check_statistics(stats, rows):
    values = np.array([[float(r['Ex_app']), float(r['Ey_app']), float(r['black_fraction'])]
                       for r in rows])
    require(np.isfinite(values).all(), 'Nonfinite metadata')
    coarse_edges = np.asarray(stats['coarse_edges'])
    fine_edges = np.asarray(stats['fine_edges'])
    fractions = np.asarray(stats['black_fraction_histograms']['edges'])
    coarse = {}
    for name, data in (('base', values[:BASE_COUNT]), ('extended', values),
                       ('added', values[BASE_COUNT:])):
        entry = stats[name]
        require(entry['n'] == len(data), f'Stale analysis count: {name}')
        for col, field in enumerate(('Ex', 'Ey', 'black_fraction')):
            for key, actual in (('minimum', data[:, col].min()),
                                ('maximum', data[:, col].max()),
                                ('median', np.median(data[:, col])),
                                ('mean', data[:, col].mean())):
                require(np.isclose(entry[field][key], actual, rtol=1e-12, atol=1e-12),
                        f'Stale analysis: {name}/{field}/{key}')
        coarse[name] = np.histogram2d(data[:, 0], data[:, 1], bins=coarse_edges)[0].T.astype(int)
        fine = np.histogram2d(data[:, 0], data[:, 1], bins=fine_edges)[0].T.astype(int)
        require(coarse[name].tolist() == entry['coarse_counts_y_rows_x_columns'],
                f'Stale coarse distribution: {name}')
        require(fine.tolist() == entry['fine_counts_y_rows_x_columns'],
                f'Stale fine distribution: {name}')
        require(np.histogram(data[:, 2], fractions)[0].tolist()
                == stats['black_fraction_histograms'][name]['counts'],
                f'Stale phase-fraction distribution: {name}')
    support = coarse['extended'] > 0
    for name in ('base', 'extended'):
        cell_counts = coarse[name][support]
        actual = float(cell_counts.std() / cell_counts.mean())
        require(np.isclose(actual, stats['coarse_count_CV_on_same_extended_observed_support'][name],
                           rtol=1e-12), f'Stale common-support CV: {name}')


def correction_impact(work, rows, review):
    """Summarize the actual final review correction without copying large selections."""
    before_path, after_path = work / 'pre_correction_selection.json', work / 'full_selection.json'
    before, after = read_json(before_path), read_json(after_path)
    before_hashes, after_hashes = set(before['selected_hashes']), set(after['selected_hashes'])
    final_rows = {r['sha256_pixels']: r for r in rows if r.get('source_mode') == 'full'
                  and r.get('generation_stage') == 'reference_style_expansion'}
    require(after_hashes == set(final_rows), 'Current full selection differs from published pixels')
    changes = review.get('postselection_visual_corrections', [])
    rejected = {r['pixels_sha256'] for r in changes if r['final_verdict'] == 'exclude'}
    require(rejected and rejected.isdisjoint(r['sha256_pixels'] for r in rows),
            'Final review correction was not applied to the published set')
    before_lookup = {r['hash']: r for r in before['selected']}
    after_lookup = {r['hash']: r for r in after['selected']}
    removed, replacements = sorted(before_hashes - after_hashes), sorted(after_hashes - before_hashes)
    require(len(removed) == len(replacements), 'Final correction changed the selected count')
    def item(digest, lookup):
        source = lookup[digest]
        result = dict(pixels_sha256=digest, source_key=source['key'],
                      style_family=source['style_family'])
        if digest in final_rows:
            result['final_filename'] = final_rows[digest]['filename']
        return result
    return dict(
        pre_correction_selection_sha256=sha(before_path),
        final_selection_sha256=sha(after_path),
        before_count=len(before_hashes), final_count=len(after_hashes),
        removed_count=len(removed), replacement_count=len(replacements),
        explicit_final_AI_exclusion_count=len(rejected & set(removed)),
        other_selection_changes_after_correction=len(set(removed) - rejected),
        removed=[dict(item(d, before_lookup), reason_category=(
            'explicit_final_AI_exclusion' if d in rejected else 'greedy_selection_changed_after_correction'))
                 for d in removed],
        replacements=[item(d, after_lookup) for d in replacements],
        review_corrections=changes,
        note='Prepublication filenames are historical; all identities use actual pixel SHA256. '
             'The compact summary omits both large selection manifests.')


def preflight(work, analysis):
    """Read and bind every published input before creating any staging output."""
    live = ROOT / 'cat_binary_dataset'
    evidence = {name: read_json(work / name) for name in EVIDENCE}
    pub, final, package = (evidence[name] for name in
                          ('publication_state.json', 'finalized_staging.json', 'package_export_check.json'))
    require(pub.get('stage') == 'published_and_validated' and pub.get('count') == COUNT,
            'Publication has not finished and validated 8000 images')
    require(Path(pub['output']).resolve() == live.resolve(), 'Publication points to another dataset')
    require(pub.get('original1250_pixels_and_png_bytes_preserved') is True
            and pub.get('original1250_sample_metadata_preserved') is True,
            'Original 1250 preservation was not validated')
    require(final.get('count') == COUNT and final.get('standalone_export_audit') == 'pass',
            'Finalization has not passed')
    require(final.get('package_audit_sha256') == sha(work / 'package_export_check.json'),
            'Finalized staging is not bound to the current package audit')
    require(package.get('count') == COUNT and package.get('all_six_deliverable_files_byte_identical')
            and package.get('all_pngs_byte_identical'), 'Standalone export audit failed')
    deliverable_hashes = {name: sha(live / name) for name in DELIVERABLES}
    require(pub.get('deliverable_sha256') == deliverable_hashes
            == package.get('staged_deliverable_sha256'), 'Published deliverable hashes differ')
    for name, digest in pub.get('preserved_user_files', {}).items():
        require(Path(name).name == name and sha(live / name) == digest,
                f'Preserved user file changed: {name}')
    rows = read_rows(live / 'metadata.csv')
    names = [f'cat_{i:04d}.png' for i in range(COUNT)]
    require([r['filename'] for r in rows] == names, 'Metadata ordering or count differs')
    require(sorted(p.name for p in (live / 'images').glob('*.png')) == names, 'PNG inventory differs')
    require(len({r['seed'] for r in rows}) == COUNT and len({r['sha256_pixels'] for r in rows}) == COUNT,
            'Source or pixel hashes are not unique')
    def row_identity(subset):
        payload = [[r['filename'], r['sha256_pixels'], str(r['seed']),
                    float(r['Ex_app']), float(r['Ey_app'])] for r in subset]
        return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
    require(row_identity(rows) == evidence['release_identity.json']['identity']
            and row_identity(rows[:BASE_COUNT]) == evidence['base_identity.json']['identity_sha256'],
            'Release or original identity evidence differs from current metadata')
    png_manifest = hashlib.sha256()
    for name in names:
        png_manifest.update(f'{name}:{sha(live / "images" / name)}\n'.encode('ascii'))
    require(png_manifest.hexdigest() == package.get('staged_ordered_png_manifest_sha256'),
            'Published PNG bytes differ from the standalone export audit')
    counts = Counter(r['style_family'] for r in rows[BASE_COUNT:])
    require(counts == Counter({family: 750 for _, family, _ in FAMILIES}), 'Expected nine strata of 750')
    require(all(r.get('generation_stage') == 'reference_style_expansion' for r in rows[BASE_COUNT:]),
            'New samples contain an unexpected generation stage')
    review = evidence['reference_visual_review.json']
    passed, excluded = set(review['passed_hashes']), set(review['excluded_hashes'])
    require(all(r['sha256_pixels'] in passed and r['sha256_pixels'] not in excluded
                for r in rows[BASE_COUNT:]), 'Final new pixels lack passed AI review')
    impact = correction_impact(work, rows, review)
    require(evidence['refinement.json'].get('checks_passed') is True
            and evidence['pilot_refinement.json'].get('checks_passed') is True, 'Mesh checks failed')
    require(evidence['reproduction.json'].get('all_exact') is True, 'GPU reproduction check failed')
    array = np.load(live / 'dataset.npy', mmap_mode='r', allow_pickle=False)
    require(array.shape == (COUNT, 224, 224) and array.dtype == np.uint8, 'Wrong NumPy array')
    with (live / 'dataset.npy').open('rb') as stream:
        stream.seek(array.offset)
        payload_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(payload_sha == evidence['refinement.json']['dataset_sha256_pixels'],
            'Mesh-refinement evidence belongs to different array pixels')
    stats = read_json(analysis / 'reference_expansion_statistics.json')
    require(stats['sources']['expanded_metadata_sha256'] == deliverable_hashes['metadata.csv'],
            'Analysis belongs to different published metadata')
    require(stats['sources']['base_metadata_sha256'] == evidence['base_identity.json']['metadata_sha256'],
            'Analysis compares a different base')
    check_statistics(stats, rows)
    preview_rows = sorted((r for r in rows if r['preview_cell'] != ''), key=lambda r: int(r['preview_cell']))
    require([int(r['preview_cell']) for r in preview_rows] == list(range(100)), 'Preview must show 100 maps')
    with Image.open(live / 'preview.png') as sheet:
        require(sheet.size == (2306, 2646), 'Unexpected preview layout')
        for cell, row in enumerate(preview_rows):
            pixels = png_pixels(live / 'images' / row['filename'])
            require(pixel_hash(pixels) == row['sha256_pixels'], 'Preview sample pixel hash mismatch')
            require(np.array_equal(pixels, array[int(row['filename'][4:8])]), 'PNG/NPY sample mismatch')
            x, y = 6 + (cell % 10) * 230, 6 + (cell // 10) * 264
            crop = np.asarray(sheet.crop((x, y, x + 224, y + 224)).convert('L'))
            require(np.array_equal(crop, pixels * 255), 'Preview tile differs from actual PNG')
    del array
    gallery = read_json(analysis / 'style_examples_index.json')
    lookup = {r['filename']: r for r in rows[BASE_COUNT:]}
    with Image.open(analysis / 'style_examples.png') as sheet:
        for k, (_, family, _) in enumerate(FAMILIES):
            group = gallery['families'][family]
            require(group['retained_count'] == 750 and len(group['examples']) == 2, 'Stale gallery counts')
            for col, entry in enumerate(group['examples']):
                require(lookup[entry['filename']]['sha256_pixels'] == entry['sha256_pixels'], 'Stale gallery index')
                x, y = 22 + (k % 3) * 484 + col * 236, 107 + (k // 3) * 314
                crop = np.asarray(sheet.crop((x, y, x + 224, y + 224)).convert('L'))
                require(set(np.unique(crop)) == {0, 255}
                        and pixel_hash(crop // 255) == entry['sha256_pixels'], 'Gallery tile differs from recorded pixels')
    for name in ANALYSIS_FILES:
        require((analysis / name).is_file(), f'Missing analysis output: {name}')
    return dict(live=live, rows=rows, preview_rows=preview_rows, evidence=evidence, stats=stats,
                deliverable_hashes=deliverable_hashes, correction_impact=impact)


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + \
           '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>'
                                           for row in rows) + '</tbody></table></div>'


def report(context):
    s, ev, rows = context['stats'], context['evidence'], context['rows']
    cv = s['coarse_count_CV_on_same_extended_observed_support']
    change = '降低（CV 上升）' if cv['extended'] > cv['base'] else '提高（CV 下降）'
    summary_rows = []
    for label, key in (('原始数据', 'base'), ('本次新增', 'added'), ('最终数据', 'extended')):
        d = s[key]
        summary_rows.append((label, d['n'], f"{d['Ex']['minimum']:.4f}–{d['Ex']['maximum']:.4f}",
                             f"{d['Ey']['minimum']:.4f}–{d['Ey']['maximum']:.4f}",
                             f"{d['black_fraction']['minimum']:.2%}–{d['black_fraction']['maximum']:.2%}",
                             d['coarse_occupied_cells'], d['fine_occupied_cells']))
    style_rows = []
    for ref, family, name in FAMILIES:
        group = [r for r in rows[BASE_COUNT:] if r['style_family'] == family]
        files = {(r['reference_filename'], r['reference_sha256']) for r in group}
        require(len(files) == 1, f'Inconsistent reference provenance: {family}')
        filename, digest = files.pop()
        style_rows.append((ref, name, len(group), filename, digest))
    fractions = s['black_fraction_histograms']
    fraction_rows = []
    for i, (lo, hi) in enumerate(zip(fractions['edges'][:-1], fractions['edges'][1:])):
        fraction_rows.append((f'{lo:.0%}–{hi:.0%}', fractions['base']['counts'][i],
                              fractions['added']['counts'][i], fractions['extended']['counts'][i],
                              f"{fractions['extended']['proportions'][i]:.2%}"))
    sparse = [(f"{r['Ex_interval'][0]:g}–{r['Ex_interval'][1]:g}",
               f"{r['Ey_interval'][0]:g}–{r['Ey_interval'][1]:g}", r['old'], r['added'], r['extended'])
              for r in s['remaining_sparse_coarse_cells_1_to_15']]
    pilot, full = ev['pilot_generation_state.json'], ev['full_generation_state.json']
    modes = Counter(r.get('source_mode') for r in rows[BASE_COUNT:])
    refinement, reproduction, pairs = ev['refinement.json'], ev['reproduction.json'], ev['final_pair_audit.json']
    gpu = ev['reproduction_details.json'].get('gpu', '未记录')
    torch_version = ev['reproduction_details.json'].get('torch_version', '未记录')
    source = sorted({r['source'] for r in rows[BASE_COUNT:]})
    protocol = rows[BASE_COUNT]
    evidence_links = ''.join(f'<li><a href="evidence/{esc(name)}">{esc(name)}</a></li>' for name in EVIDENCE)
    evidence_links += '<li><a href="evidence/final_correction_impact.json">final_correction_impact.json</a></li>'
    correction = context['correction_impact']
    support = s['same_property_region_style_support']
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>8,000 张 Binary Cat 数据集 · 进度与审核报告</title><style>
body{{margin:0;background:#eef2f6;color:#182a3b;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1120px;margin:auto;padding:36px 28px 70px;background:white}}h1{{font-size:30px;line-height:1.35}}h2{{margin-top:38px;border-bottom:2px solid #dde7ef;padding-bottom:8px}}
a{{color:#165d91}}.lead{{font-size:18px}}.notice{{padding:18px 22px;background:#fff4df;border-left:5px solid #c68a28}}.muted{{color:#536473;font-size:14px}}
img{{max-width:100%;height:auto;border:1px solid #dbe3e9}}.binary{{image-rendering:pixelated}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 10px;border:1px solid #dbe3e9;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#edf4f9}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere;background:#eff3f6;padding:2px 5px}}.columns{{columns:2}}@media(max-width:700px){{main{{padding:22px 14px}}.columns{{columns:1}}}}
</style></head><body><main>
<p class="muted">本地审核材料 · {stamp} · 解压 ZIP 后打开本文件；图片和证据使用包内相对链接。</p>
<h1>Binary Cat Mechanical Metamaterials<br>8,000 张数据集进度与审核报告</h1>
<p class="lead">已发布并验证 <strong>{len(rows):,} 张</strong> 224 × 224 binary 材料图。原始 {BASE_COUNT:,} 张的像素、PNG bytes、seed 与逐样本力学标签保留；新增 {len(rows)-BASE_COUNT:,} 张为独立新 sources，编号 <code>cat_1250.png–cat_7999.png</code>，9 个 reference strata 各 750 张。</p>
<div class="notice"><strong>本轮主要增加了参考风格与内部材料结构，不能把它等同于扩大力学边界。</strong>
相对原始 1,250 张，新增占据的 0.5 × 0.5 模量格为 <strong>{s['newly_observed_fine_cells']}</strong> 个；
相同已观察 support 上的 1.5 × 1.5 格计数 CV 从 <strong>{cv['base']:.3f}</strong> 变为 <strong>{cv['extended']:.3f}</strong>，均匀程度{change}。
这不是训练结果；没有训练 conditional flow matching model，也没有划分 train/test。</div>
<h2>做了什么</h2>
<ol><li>保留旧 1,250 张作为 immutable base；按用户参考图新增 9 种风格，排除 ref05 的旧 plain silhouette。</li>
<li>先生成 {pilot['source_count']} 个独立 pilot sources，自动筛选通过 {pilot['valid_sources']} 个，再依据 FEM、重复检查和实际 AI 图审选定 {modes.get('pilot',0)} 张 pilot。</li>
<li>全量生成 {full['source_count']:,} 个独立 sources，自动筛选通过 {full['valid_sources']:,} 个；完成像素绑定的 AI contact-sheet review、实际 FEM 标签及去重后保留 {modes.get('full',0):,} 张。</li>
<li>进行 224→448 mesh refinement 抽查、本机完整 source batches 再生成抽查、脱离项目 helper 文件的 standalone export 审核，随后发布并逐文件校验。</li></ol>
<p>最终发布前的再次 AI 图审明确排除 {correction['explicit_final_AI_exclusion_count']} 个此前通过的 source；重新运行 greedy selection 后，共换出 {correction['removed_count']} 个、替补 {correction['replacement_count']} 个，其中 {correction['other_selection_changes_after_correction']} 个换出由选择联动产生。相关 pixel SHA 已从最终集合移除，selection、export、mesh/QC、analysis 和发布检查均采用更新后的集合。<a href="evidence/final_correction_impact.json">查看简洁的最终复核替换记录</a>；记录中的旧编号属于发布前 staging，不能用作当前编号。</p>
<p>每个新 source 最终只保留一张；没有用旧图旋转、平移、缩放来填充数量。黑白 background、framing、身体几何、portrait/whole-body、内部纹理及 pixel grid 分层参与生成。语义与 style scores 只作为筛选辅助，metadata 中的 pose/style 是生成意图，不是人工真值分类。</p>
<h2>实际覆盖与分布</h2>
{table(['范围','数量','Ex / Esoft','Ey / Esoft','黑色硬相比例','1.5格占据数 / 36','0.5格占据数 / 324'],summary_rows)}
<p>CV = 格计数的标准差 / 均值；这里把两个数据集放在最终数据已占据的<strong>同一组 coarse cells</strong>上比较，旧数据未占据的格记为零。它依赖 bin 尺寸，只描述采样分布，不证明连续可达区域或模型生成精度。新增点超出原始 Ex/Ey 轴范围的数量为 <strong>{s['nearest_added_to_base']['added_outside_original_axis_aligned_range']}</strong>。</p>
<a href="analysis/reference_expansion_comparison.png"><img src="analysis/reference_expansion_comparison.png" alt="原1250与最终8000的Ex/Ey联合分布、边际直方图、黑色比例及风格比较"></a>
<p><a href="analysis/reference_expansion_comparison.pdf">查看可放大的 PDF</a> · <a href="analysis/reference_expansion_statistics.json">完整数值统计 JSON</a></p>
<p>在 1.5 × 1.5 的 coarse cells 中，{support['occupied_cells_with_at_least_3_reference_families']} 个格包含至少 3 种新增风格，{support['occupied_cells_with_all_9_reference_families']} 个格包含全部 9 种。落在同一格并不表示 Ex/Ey 完全相同。</p>
<h3>仍需补充的已观察稀疏区域</h3>
{table(['Ex 区间','Ey 区间','原始数','新增数','最终数'],sparse) if sparse else '<p>没有计数为 1–15 的已占据 coarse cell。</p>'}
<p>建议下一轮在这些区域、强 anisotropy 与高模量尾部采用 FEM 驱动的定向候选搜索；同时保留猫辨识和风格要求。不能直接承诺必要外边界内的任意点都有可实现猫形。训练时可使用条件分层抽样/权重，但不能替代缺失区域的新数据。</p>
<details><summary>黑色比例完整分布（每档 5%；左闭右开，最后一档包含 100%）</summary>
{table(['黑色硬相比例','原始1250','新增6750','最终8000','最终占比'],fraction_rows)}</details>
<h2>实际图像与参考风格</h2>
<a href="analysis/style_examples.png"><img class="binary" src="analysis/style_examples.png" alt="9个新风格各两张实际保留的224像素猫材料图"></a>
<p>Gallery 的 18 张是实际保留的 PNG pixels，未重新生成或美化。此处展示代表样本，不替代全量质量审查。<a href="analysis/style_examples_index.json">查看 gallery 对应编号、seed、pixel hash 与选择记录</a>。</p>
{table(['Ref','风格 stratum','新增数量','用户参考文件名','参考文件 SHA256'],style_rows)}
<p>参考图只作为 style inspiration 与文字描述来源，没有复制参考 pixels；未从网络收集猫图。ref11 采用粗 tabby 区域和内部对比结构，未添加月亮、星星等背景。参考文件的授权范围没有在此另行审查；模型来源和许可证记录见 <a href="dataset/README.md">原始 README</a>。</p>
<h3>最终随机 100 张 preview</h3>
<a href="dataset/preview.png"><img class="binary" src="dataset/preview.png" alt="最终数据随机100张10乘10预览"></a>
<p>包内 <code>preview_samples/</code> 附这 100 张的原始 binary PNG，<a href="preview_samples/index.csv">index.csv</a> 记录 preview_cell、文件名与标签，可独立加载抽查。PNG 显示黑白为 0/255；转换成 bool 再转 uint8 得到 0/1。</p>
<h2>生成与 FEM 协议</h2>
<p>本地 diffusion source：<code>{esc('; '.join(source))}</code>；512px、9 inference steps，master seed {pilot['seed']}、batch size {pilot['batch_size']}，确定性 binary preprocessing 输出 224 × 224。GPU 抽查记录：<code>{esc(gpu)}</code>，PyTorch <code>{esc(torch_version)}</code>。SigLIP 语义/style screening 使用固定 revision；原始分数及阈值保留在 metadata。</p>
{table(['项目','固定设定'],[
('像素与材料','0 = 黑色硬实体；1 = 白色软实体。白色不是孔洞；不做周期阵列。'),
('材料参数',f"Esoft={protocol['E_soft']}，Ehard={protocol['E_hard']}；νsoft={protocol['nu_soft']}，νhard={protocol['nu_hard']}"),
('试样与力学模型',f"{protocol['specimen_width_mm']} × {protocol['specimen_height_mm']} × {protocol['specimen_thickness_mm']} mm；小应变 linear elasticity、perfect bonding、Q4 plane stress"),
('边界条件',f"独立有限试样；x/y分别施加名义拉伸应变 {protocol['nominal_strain']}，侧边 traction-free；约束刚体平移。由反力计算 apparent specimen Ex/Ey。"),
('网格与求解','每 pixel 一个 Q4 element，224×224；double precision AMG-CG，记录 residual、energy/reaction 与 force balance。'),
('协议hash',protocol['solver_protocol_sha256'])])}
<p>数值模量以 Esoft=1 归一化。标签只适用于记录的 specimen / boundary conditions；不能直接解释成任意尺寸、周期均匀化或打印实测模量。共同 Poisson ratio 下检查 fixed-fraction 必要界：<code>1/(1−0.9f) ≤ Ex,Ey ≤ 1+9f</code>；通过此检查不代表猫形可达域已覆盖。</p>
<h2>质量验证与可追溯证据</h2>
{table(['检查','实际记录'],[
('技术约束','发布验证通过：8000×224×224、uint8、仅0/1；PNG/NPY/CSV filename顺序及pixel SHA匹配；无全黑/全白图。'),
('旧数据保留','旧1250的array pixels与PNG bytes、逐样本metadata保留；property_bin_count / balanced_sampling_weight / preview_cell为全数据全局字段，已重算。'),
('精确重复/独立sources',f"新增相关 exact pixel duplicates={pairs['exact_pixel_duplicate_pairs_involving_new']}；重复source seeds={pairs['repeated_source_seeds_involving_new']}。"),
('Near-duplicate筛查',f"按公开数值规则，新增相关 mechanical conflicts={pairs['mechanical_near_conflicts_remaining']}；strict visual conflicts={pairs['strict_visual_conflicts_remaining']}。旧base-base情况另见base_pair_audit。"),
('Mesh refinement',f"full抽查{refinement['count']}张：224→448最大相对变化{refinement['maximum_relative_change']:.3%}；224标签重算最大相对误差{refinement['maximum_csv_recomputation_relative_error']:.3g}；checks_passed={refinement['checks_passed']}。"),
('本机再生成抽查',f"{reproduction['batches']}个完整batch、{reproduction['checked_sources']}个sources，覆盖{len(reproduction['retained_reference_families'])}种新风格；source mask与binary pixels全部精确一致={reproduction['all_exact']}。"),
('Standalone export','独立目录运行embedded code + embedded review；6个deliverables与8000PNG byte-identical。此audit复用缓存/FEM标签，不是全量重新GPU生成。'),
('本审核包','发布hash链、全部PNG byte manifest、统计metadata绑定、gallery pixels及preview100张重新检查；ZIP CRC与包内SHA manifest另行校验。')])}
<p>AI review 由实际 contact-sheet 逐格查看及疑难放大完成，判断绑定到实际 pixel SHA；<strong>没有声称 human visual inspection，也不保证绝对辨识正确</strong>。Near-duplicate 阈值是明确的数值规则，不等于每两张图对人眼都明显不同。Mesh refinement 是选定样本检查，不是全部8000张的 continuum error 证书；GPU bitwise 抽查仅限本机配置。</p>
<details><summary>原始 QC / 发布证据文件</summary><ul class="columns">{evidence_links}</ul></details>
<h2>审核包内容与使用</h2>
<p>完整数据仍在项目原目录 <code>{esc(context['live'].resolve())}</code>。本 ZIP 是<strong>审核材料包</strong>，不含约400MB的 <code>dataset.npy</code>、全部8000张PNG、模型权重、候选cache或大量QA页。包内 metadata 是完整8000行；完整数组可按相同编号在原目录读取。</p>
<ul><li><a href="dataset/quality_report.md">quality_report.md</a> 与 <a href="dataset/README.md">README.md</a>：实际发布版本，原bytes保留。</li>
<li><a href="dataset/metadata.csv">metadata.csv</a>：完整标签、seed、source、reference hash、QC。64-bit seed 请当字符串读取，避免电子表格损失精度。</li>
<li><a href="dataset/generate_dataset.py">generate_dataset.py</a>：自包含 generation / preprocessing / FEM / selection pipeline；重现仍需 README 依赖与模型checkpoint。</li>
<li><code>analysis/scripts/</code>：实际比较与gallery脚本。比较脚本只需metadata；gallery完整重现需原目录PNG。</li>
<li><a href="SHA256SUMS.txt">SHA256SUMS.txt</a>：除清单本身以外每个包内文件的实际SHA256；<code>python verify_package.py</code> 检查解压文件。SHA用于完整性核对，不是第三方数字签名。</li></ul>
<p class="muted">没有训练模型，没有 train/test split，没有宣称打印验证或实验测量；未给出未经计算的性能标签。后续 conditional generation 的可达精度需要独立训练与 FEM 复验。</p>
</main></body></html>'''


VERIFY_SCRIPT = '''"""Verify the extracted review package using only the Python standard library."""
from pathlib import Path, PurePosixPath
import hashlib
root=Path(__file__).resolve().parent
seen=set()
for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    digest,name=line.split('  ',1)
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\\\' in name or ':' in name:
        raise ValueError('Unsafe manifest path')
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Path escapes package')
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=digest:raise ValueError('SHA mismatch: '+name)
    seen.add(name)
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
if actual!=seen|{'SHA256SUMS.txt'}:raise ValueError('Unexpected/missing package files')
print('Verified',len(seen),'payload files; all SHA256 values match.')
'''


def write_manifest(stage):
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    entries = {safe_member(p.relative_to(stage).as_posix()): sha(p) for p in files}
    (stage / 'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name, digest in entries.items()),
                                       encoding='utf-8', newline='\n')
    return entries


def validate_zip(path, entries):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive entries')
        require(set(names) == set(entries) | {'SHA256SUMS.txt'}, 'Unexpected archive content')
        for name in names:
            safe_member(name)
        require(archive.testzip() is None, 'ZIP CRC check failed')
        for name, expected in entries.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    f'Archive SHA differs: {name}')
        actual_manifest = archive.read('SHA256SUMS.txt').decode('utf-8')
        expected_manifest = ''.join(f'{digest}  {name}\n' for name, digest in entries.items())
        require(actual_manifest == expected_manifest, 'Archive manifest differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--analysis-dir', type=Path, default=ROOT / 'analysis' / 'reference_expansion')
    parser.add_argument('--replace', action='store_true', help='Replace an existing review ZIP after full checks')
    args = parser.parse_args()
    work, analysis = args.work_dir.resolve(), args.analysis_dir.resolve()
    output = ROOT / 'cat_dataset_review_8000.zip'
    require(args.replace or not output.exists(), f'ZIP already exists; use --replace explicitly: {output}')
    require(not work.is_relative_to(ROOT.resolve()), 'Scratch work-dir must be outside the project')
    context = preflight(work, analysis)
    staging_root = work / 'review_package'
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='build-', dir=staging_root))
    plan = {}
    for name in DELIVERABLES:
        if name != 'dataset.npy':
            plan[f'dataset/{name}'] = context['live'] / name
    for name in ANALYSIS_FILES:
        plan[f'analysis/{name}'] = analysis / name
    for name in ANALYSIS_SCRIPTS:
        plan[f'analysis/scripts/{name}'] = ROOT / 'analysis' / name
    for name in EVIDENCE:
        plan[f'evidence/{name}'] = work / name
    for row in context['preview_rows']:
        plan[f'preview_samples/{row["filename"]}'] = context['live'] / 'images' / row['filename']
    copied_hashes = {}
    for name, source in plan.items():
        safe_member(name)
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, destination)
        require(before == sha(destination) == sha(source), f'Input changed while copying: {name}')
        copied_hashes[name] = before
    fields = ('preview_cell', 'filename', 'sha256_pixels', 'seed', 'Ex_app', 'Ey_app', 'black_fraction')
    with (stage / 'preview_samples' / 'index.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(context['preview_rows'])
    (stage / 'START_HERE.html').write_text(report(context), encoding='utf-8')
    (stage / 'verify_package.py').write_text(VERIFY_SCRIPT, encoding='utf-8')
    (stage / 'evidence' / 'final_correction_impact.json').write_text(
        json.dumps(context['correction_impact'], ensure_ascii=False, indent=2), encoding='utf-8')
    entries = write_manifest(stage)
    require(all(entries[name] == value for name, value in copied_hashes.items()), 'Staged copy changed')
    # Rebind publication and metadata after all copies, before publishing the ZIP.
    require(sha(context['live'] / 'metadata.csv') == context['deliverable_hashes']['metadata.csv'],
            'Live metadata changed during packaging')
    require(sha(work / 'publication_state.json') == copied_hashes['evidence/publication_state.json'],
            'Publication record changed during packaging')
    temporary = staging_root / (stage.name + '.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in stage.rglob('*') if p.is_file()):
            archive.write(path, arcname=safe_member(path.relative_to(stage).as_posix()))
    validate_zip(temporary, entries)
    # Replace is limited to this fixed review ZIP; never touches dataset or analysis.
    os.replace(temporary, output)
    validate_zip(output, entries)
    print(json.dumps(dict(zip=str(output),bytes=output.stat().st_size,sha256=sha(output),
                          payload_files=len(entries),preview_pngs=100,complete_dataset_included=False,
                          zip_crc='pass',manifest_sha256='pass',staging=str(stage)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
