"""Add original-label counterparts of the sixteen curated style scatter plots."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_hex

from plot_curated_augmented import style_axes, in_envelope
from common import to_sa, sha_file

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/'cat_binary_dataset_curated_v1'
OUT = ROOT/'analysis/curated_augmented_distribution'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def scatter(space, style, points, color):
    fig, ax = style_axes(space)
    artist = ax.scatter(points[:,0],points[:,1],s=8,c=color,alpha=.34,linewidths=0,rasterized=True,zorder=3)
    np.testing.assert_array_equal(np.asarray(artist.get_offsets()),points)
    fig.text(.105,.947,f'{style["group_id"]}  {style["group_name_zh"]}',fontsize=20,color=color)
    fig.text(.105,.912,style['group_name_en']+' | Joint distribution',fontsize=14)
    fig.text(.105,.873,f'{len(points):,} original images = {len(points):,} points | No augmentation',fontsize=10.7,color='#536373')
    fig.text(.105,.090,'All curated train / validation / test samples pooled; each image plotted once.',fontsize=9,color='#536373')
    fig.text(.105,.064,'Exact original FEM labels; no jitter, rotation, mirror or Ex/Ey exchange.',fontsize=9,color='#536373')
    fig.text(.105,.038,'Gray: outside necessary material bounds. White is not proof of cat-shape feasibility.',fontsize=8.7,color='#617080')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.texts+[ax.xaxis.label,ax.yaxis.label]:
        box=text.get_window_extent(renderer)
        assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height, text.get_text()
    path=OUT/space/f'{style["group_id"]}_{style["group_label"]}_no_augmentation.png'
    fig.savefig(path,dpi=200,facecolor='white');plt.close(fig)
    return path


def gallery(classes):
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>八类原始物性分布 / No augmentation</title>',
        '<style>body{font:16px/1.6 Segoe UI,Microsoft YaHei,sans-serif;background:#eef2f5;color:#253342;margin:24px}a{color:#2465a0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0;background:white;padding:8px;border-radius:8px}img{width:100%;height:auto}figcaption{padding:5px}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>',
        '<h1>八类原始物性分布：No augmentation</h1><p>7,857张原图；每个坐标系中每张图只绘制一次。使用真实FEM标签，不加jitter，不旋转、镜像或交换Ex/Ey。颜色、坐标范围和必要物理边界沿用各类原有散点图。</p>',
        '<p>配套6×6未增强热图：<a href="grid_v1_equal_width/ex_ey/09_all_styles_counts_no_augmentation.png">Ex/Ey</a> · <a href="grid_v1_equal_width/s_a/09_all_styles_counts_no_augmentation.png">s/a</a></p>']
    for space,label in [('ex_ey','Ex/Ey'),('s_a','s/a')]:
        parts.append(f'<h2>{label}</h2><div class="grid">')
        for c in classes:
            path=f'{space}/{c["group_id"]}_{c["group_label"]}_no_augmentation.png'
            caption=c['group_id']+' '+c['group_name_zh']+f' · {c["count"]:,}张'
            parts.append(f'<figure><a href="{path}"><img src="{path}" alt="{label} {caption}"></a><figcaption>{caption}</figcaption></figure>')
        parts.append('</div>')
    parts.append('</html>')
    (OUT/'index_no_augmentation.html').write_text('\n'.join(parts),encoding='utf-8')


def main():
    classes=read(DATA/'classes.json')['classes']
    protected=[DATA/'metadata.csv',DATA/'data.npz',OUT/'index.html',ROOT/'analysis/plot_curated_augmented.py']
    for space in ['ex_ey','s_a']:
        protected += [OUT/space/f'{c["group_id"]}_{c["group_label"]}.png' for c in classes]
        protected.append(OUT/space/'counts.json')
        protected += list((OUT/'grid_v1_equal_width'/space).glob('*.png'))
        protected += list((OUT/'grid_v1_equal_width'/space).glob('*.json'))
    before={p.relative_to(ROOT).as_posix():sha_file(p) for p in protected}
    with (DATA/'metadata.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    ids=np.array([r['new_id'] for r in rows]);group_ids=np.array([r['group_id'] for r in rows])
    physical=np.array([[float(r['Ex_app']),float(r['Ey_app'])] for r in rows])
    soft=np.array([float(r['E_soft']) for r in rows])
    assert len(rows)==len(set(ids))==7857 and np.all(soft>0)
    xy=physical/soft[:,None];sa=to_sa(xy[:,0],xy[:,1])
    assert in_envelope(xy).all()
    with np.load(DATA/'data.npz',allow_pickle=False) as frozen:
        np.testing.assert_array_equal(ids,frozen['sample_ids'])
        np.testing.assert_array_equal(physical,frozen['moduli'])
        np.testing.assert_allclose(sa,frozen['sa'],rtol=1e-13,atol=1e-14)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.labelsize':12,
        'axes.spines.top':False,'axes.spines.right':False,'text.color':'#253342',
        'axes.labelcolor':'#253342','axes.edgecolor':'#617080','axes.unicode_minus':False,'mathtext.fontset':'dejavusans'})
    palette=plt.get_cmap('tab10')
    colors={c['group_id']:to_hex(palette(i)) for c,i in zip(classes,[0,1,2,3,4,5,6,9])}
    original_colors={s['group_id']:s['color'] for s in read(OUT/'statistics.json')['styles']}
    assert colors==original_colors
    figures=[]
    for c in classes:
        chosen=group_ids==c['group_id'];assert int(chosen.sum())==c['count']
        for space,points in [('ex_ey',xy),('s_a',sa)]:
            path=scatter(space,c,points[chosen],colors[c['group_id']])
            figures.append(dict(path=path.relative_to(OUT).as_posix(),space=space,group_id=c['group_id'],
                                points=int(chosen.sum()),color=colors[c['group_id']],sha256=sha_file(path)))
    assert len(figures)==16
    for space in ['ex_ey','s_a']:
        assert sum(f['points'] for f in figures if f['space']==space)==7857
    np.savez_compressed(OUT/'plot_data_no_augmentation.npz',sample_ids=ids,group_id=group_ids,
                        Ex_over_Esoft=xy[:,0],Ey_over_Esoft=xy[:,1],s=sa[:,0],a=sa[:,1])
    gallery(classes)
    assert all(sha_file(ROOT/p)==digest for p,digest in before.items())
    manifest=dict(source_dataset=DATA.relative_to(ROOT).as_posix(),source_images=7857,number_of_figures=16,
                  population='All curated original images across train/validation/test; one point per source in each coordinate system.',
                  augmentation=False,jitter=False,weighted_sampling=False,exact_scatter_coordinates_verified=True,
                  text_bounds_verified=True,source_labels_and_ids_verified=True,plots=figures,
                  protected_files_sha256=before,protected_files_unchanged=True,
                  plot_data_sha256=sha_file(OUT/'plot_data_no_augmentation.npz'),
                  command='.\\cfm_v0\\.venv\\Scripts\\python.exe analysis\\plot_curated_unaugmented.py',
                  script_sha256=sha_file(Path(__file__)))
    (OUT/'unaugmented_figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(figures),source_images=len(rows),points_per_coordinate_system=7857,
                         preserved_existing_files=len(before),gallery=str(OUT/'index_no_augmentation.html')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
