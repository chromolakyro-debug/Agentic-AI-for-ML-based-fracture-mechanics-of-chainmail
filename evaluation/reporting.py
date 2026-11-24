import html,json,shutil
import numpy as np
from utils import HERE,read,save,csvwrite,sha,digest,distance_text

def plots(run,rows,summary):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run.out/'figures';out.mkdir(exist_ok=True);grid=run.grid
    for space in ['s_a','Ex_Ey']:
        fig,ax=plt.subplots(figsize=(9,7));x,y=('s','a') if space=='s_a' else ('Ex','Ey')
        allx=[];ally=[]
        for t in run.targets:
            tx,ty=t[x+'_target'],t[y+'_target'];allx.append(tx);ally.append(ty)
            rr=[r for r in rows if r['target_id']==t['target_id'] and r['status']=='ok']
            for r in rr:
                px,py=r[x+'_FEM'],r[y+'_FEM'];ax.plot([tx,px],[ty,py],color='steelblue',lw=.55,alpha=.22);allx.append(px);ally.append(py)
            ax.scatter([r[x+'_FEM'] for r in rr],[r[y+'_FEM'] for r in rr],s=14,c='#226cba',alpha=.72,zorder=3)
            ax.scatter(tx,ty,marker='*',s=110,c='#f2ab27',edgecolors='#4c3600',linewidths=.5,zorder=4);ax.annotate(t['target_id'],(tx,ty),xytext=(5,5),textcoords='offset points',fontsize=8,zorder=5)
        if space=='s_a':
            for v in grid['s_edges']:ax.axvline(v,color='.75',lw=.6,zorder=0)
            for v in grid['a_edges']:ax.axhline(v,color='.75',lw=.6,zorder=0)
            allx+=grid['s_edges'];ally+=grid['a_edges']
        else:
            soft=run.grid['esoft'];hard=run.cfg['fem']['parameters']['hard_modulus'];allx+=[soft,hard];ally+=[soft,hard];ax.grid(alpha=.2)
        for vals,setlim in [(allx,ax.set_xlim),(ally,ax.set_ylim)]:
            lo,hi=min(vals),max(vals);pad=max(.02*(hi-lo),.02);setlim(lo-pad,hi+pad)
        ax.set(xlabel=x,ylabel=y,title=f'{run.cfg["run_id"]}: targets (*) and actual FEM (blue)'+(' — DIAGNOSTIC ONLY' if run.diagnostic else ''))
        fig.tight_layout();fig.savefig(out/f'overview_{space}.png',dpi=170);plt.close(fig)
    if all(t['grid_row'] is not None for t in run.targets):
        for metric,title in [('mean_d_mech','Mean mechanical error distance'),('var_d_mech','Sample variance of distance (ddof=1)')]:
            values=np.full((len(grid['a_edges'])-1,len(grid['s_edges'])-1),np.nan)
            for r in summary:
                if r[metric] is not None:values[r['grid_row']-1,r['grid_col']-1]=r[metric]
            fig,ax=plt.subplots(figsize=(10,8));cmap=plt.get_cmap('magma').copy();cmap.set_bad('white')
            mesh=ax.pcolormesh(grid['s_edges'],grid['a_edges'],np.ma.masked_invalid(values),cmap=cmap,shading='flat');fig.colorbar(mesh,ax=ax,label=metric)
            for r in summary:
                row,col=r['grid_row'],r['grid_col'];v=r[metric];label=f'{v:.3g}' if v is not None else 'undefined'
                ax.text(np.mean(grid['s_edges'][col-1:col+1]),np.mean(grid['a_edges'][row-1:row+1]),f'{r["target_id"]}\n{label}\nn={r["n_valid"]}/{r["n_total"]}',ha='center',va='center',fontsize=8,
                    bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=1))
            ax.set(xlabel='s',ylabel='a',title=title);ax.set_xticks(grid['s_edges']);ax.set_yticks(grid['a_edges']);ax.grid(alpha=.25)
            fig.text(.5,.015,distance_text(run.distance),ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/f'{metric}.png',dpi=170);plt.close(fig)
    # Fixed seeds, all diagnostic targets; never pick visually attractive samples.
    chosen=rows if run.diagnostic else [r for r in rows if r['seed'] in run.seeds[:2]]
    ncols=10 if run.diagnostic else 8;nrows=(len(chosen)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*1.8,nrows*2.5),squeeze=False)
    from PIL import Image
    for ax in axes.flat:ax.axis('off')
    for ax,r in zip(axes.flat,chosen):
        ax.imshow(Image.open(run.out/r['image']),cmap='gray',vmin=0,vmax=255)
        actual=f'{r["Ex_FEM"]:.2f}, {r["Ey_FEM"]:.2f}' if r['status']=='ok' else 'FEM FAILED'
        ax.set_title(f'{r["target_id"]} / {r["seed"]}\nT {r["Ex_target"]:.2f}, {r["Ey_target"]:.2f}\nF {actual}',fontsize=8)
    fig.suptitle(run.cfg['run_id']+' | fixed seeds, unfiltered binary FEM inputs'+(' | DIAGNOSTIC ONLY' if run.diagnostic else ''),fontsize=12)
    fig.subplots_adjust(left=.015,right=.99,bottom=.015,top=.94,wspace=.1,hspace=.43)
    fig.savefig(out/'fixed_examples.png',dpi=150);plt.close(fig)

def report(run):
    rows=read(run.out/'results.json');summary=read(run.out/'summary.json');ar=read(run.out/'target_retrieval.json');br=read(run.out/'fem_retrieval.json')
    plots(run,rows,summary)
    assets=run.out/'retrieval_images';assets.mkdir(exist_ok=True)
    for q in ar+br:
        for r in q['neighbors']:
            src=run.dataset.paths[r['dataset_index']];dst=assets/f'{r["dataset_index"]:06d}{src.suffix}'
            if not dst.exists() or sha(dst)!=r['png_sha256']:shutil.copyfile(src,dst)
            if sha(dst)!=r['png_sha256']:raise ValueError('Retrieval image mismatch')
            r['display_image']=dst.relative_to(run.out).as_posix()
    payload=dict(run_id=run.cfg['run_id'],diagnostic_only=run.diagnostic,targets=run.targets,seeds=run.seeds,rows=rows,summary=summary,
        target_retrieval=ar,fem_retrieval=br,distance=run.distance,distance_formula=distance_text(run.distance),tolerances=run.cfg['tolerances'])
    save(run.out/'report_data.json',payload)
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    template=PathTemplate
    (run.out/'index.html').write_text(template.replace('__REPORT_DATA__',data),encoding='utf-8')
    # This JSON is the only source of numerical labels in HTML. Verify its lossless round trip.
    from verify_report import verify
    audit=verify(run.out);save(run.out/'report_verification.json',audit)
    ok=[r for r in rows if r['status']=='ok'];total=len(rows);hits=sum(r['hit_s_a'] for r in ok);hxy=sum(r['hit_Ex_Ey'] for r in ok)
    table='\n'.join(f'|{r["target_id"]}|{r["n_valid"]}/{r["n_total"]}|{r["mean_d_mech"]:.5f}|{r["var_d_mech"]:.5f}|' if r['mean_d_mech'] is not None and r['var_d_mech'] is not None else f'|{r["target_id"]}|{r["n_valid"]}/{r["n_total"]}|未定义|未定义|' for r in summary)
    report=f'''# Evaluation {run.cfg['run_id']}

{'本次仅为模块验证，使用4个既有development targets。用户红圈24格尚未确定，不能视作正式评估。' if run.diagnostic else '采用独立冻结的24个用户选定网格中点及10个固定seeds；未训练或修改模型。'}

Checkpoint：`{run.cfg['model']['checkpoint']}`，`{run.cfg['model']['weights_key']}`权重；hash见manifest。数据版本`{run.cfg['dataset']['version']}`，共{len(run.dataset.rows)}张，其中{len(run.dataset.train)}张训练图用于retrieval。实际模型信息及训练augmentation见model_info.json；本次按用户要求在检索距离中比较原轴序和交换轴序，选择较小值，每个来源只占一个名额。

定义为`s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]`、`a=log(Ex/Ey)`，`Esoft=1`。模量沿用原FEM单位。距离为`{distance_text(run.distance)}`；固定尺度不由生成结果拟合。原输出保留float32；阈值`raw>=0`为白色软材料1，黑色硬材料0。FEM对保存的二值数组逐图求解，HTML使用同一材料场，无插值、缩放或反色。

实际完成{total}张生成，FEM有效{len(ok)}/{total}，失败{total-len(ok)}。Ex/Ey各相对误差≤5%的联合命中为{hxy}/{total}；`|Δs|≤log(1.05)`且`|Δa|≤log(1.10)`为{hits}/{total}。失败计入分母。FEM residual、反力和能量检查保留在results.json/csv中。

|Target|有效数|mean(d_mech)|var(d_mech), ddof=1|
|---|---:|---:|---:|
{table}

`index.html`按target展示固定10张生成、target retrieval和最佳生成图FEM retrieval，均不进行外观筛选。第三行query是按FEM选择的best-of-10，不能代表随机生成平均水平。图中“original”是显示图像原始轴序模量，“matched”是用于距离的轴序；交换方向时仍显示原训练图。旋转/镜像按钮仅改变查看角度，原始FEM值与下载数组不变。

猫形辨识、外观多样性及memorization留给人工审阅；机械近邻不等于视觉近邻。页面提供逐张人工评级及JSON导出。不能仅凭像素不同宣称新颖结构。正式24格缺失时不绘制假冒正式热图；热图实现及样本方差/缺失值逻辑另有核心测试。

模型/数据/划分绑定、所有图像内容、targets、seeds、采样设置、FEM设置及代码hash见manifest和code_hashes。原始数据、metadata、split和checkpoint未更改。复现入口和迁移说明见evaluation/README.md。
'''
    (run.out/'REPORT.md').write_text(report,encoding='utf-8')

PathTemplate='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cat mechanical evaluation</title>
<style>body{font-family:system-ui,sans-serif;margin:22px;color:#19242f;background:#f6f7f8}h1{font-size:24px}h2{font-size:18px}p{line-height:1.6}section{background:white;padding:16px;margin:25px 0;border:1px solid #ccd4da;border-radius:8px}.row{display:grid;grid-template-columns:repeat(10,minmax(130px,1fr));gap:8px;overflow-x:auto}.card{min-width:130px;font-size:11px;line-height:1.45;border:1px solid #dde1e5;padding:5px;word-break:break-word}.picture{width:100%;aspect-ratio:1;background:white;overflow:hidden}.picture img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}.best{border:2px solid #ce9000}.buttons{display:flex;gap:3px;flex-wrap:wrap}button,select{font-size:11px;padding:3px;cursor:pointer}code{background:#eef1f4;padding:3px}.warning{background:#fff1cc;padding:12px}.metric{font-family:ui-monospace,monospace}.foot{color:#536574}textarea{width:95%;height:50px}nav a{margin:0 8px}</style></head>
<body><h1 id="title"></h1><div id="intro"></div><p class="metric" id="distance"></p><p>Generated显示固定全体seeds，FEM针对其保存的二值像素独立求解。第三行按真实FEM距离选择best-of-10作为query，并非普通随机生成质量。检索仅在对应训练集中进行；原方向和交换方向取较小距离，来源ID去重。</p><p>Retrieval显示未旋转原图：original是原图标签，matched是计入距离的轴序。旋转↺及左右镜像仅用于视觉比较；显示的FEM始终对应保存图的原始全局坐标。没有视觉近邻或自动猫形判定。</p>
<nav><a href="results.csv">逐样本CSV</a><a href="summary.csv">统计CSV</a><a href="manifest.json">Manifest</a><a href="figures/overview_s_a.png">s–a连线</a><a href="figures/overview_Ex_Ey.png">Ex–Ey连线</a><a href="REPORT.md">中文报告</a></nav><p><button onclick="downloadRatings()">导出人工评级JSON</button> 评级需手动导出，不发送到服务器。</p><div id="panels"></div>
<script id="report-data" type="application/json">__REPORT_DATA__</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent),ratings={};
const f=x=>x==null?'N/A':Number(x).toFixed(5),esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
document.getElementById('title').textContent='Evaluation: '+D.run_id;
document.getElementById('distance').textContent='d_mech及retrieval距离: '+D.distance_formula+'; 方差ddof=1。';
document.getElementById('intro').innerHTML=D.diagnostic_only?'<p class="warning">DIAGNOSTIC ONLY：4个既有development targets，不是用户要求的24格正式评估。红圈位置缺失。</p>':'<p>24个冻结target × 10个固定seeds。</p><p><a href="figures/mean_d_mech.png">mean(d_mech)热图</a> · <a href="figures/var_d_mech.png">var(d_mech)热图</a></p>';
function view(btn,action){let c=btn.closest('.card'),im=c.querySelector('img');let k=+(im.dataset.k||0),m=+(im.dataset.m||1);if(action==='r')k=(k+1)%4;if(action==='f')m=-m;if(action==='reset'){k=0;m=1}im.dataset.k=k;im.dataset.m=m;im.style.transform=`scaleX(${m}) rotate(${-90*k}deg)`;c.querySelector('.viewstate').textContent=`仅显示: CCW ${90*k}°, mirror ${m<0}`;}
function generated(r,best){return `<div class="card ${r.sample_key===best?'best':''}" data-key="${r.sample_key}"><div class="picture"><img src="${r.image}" alt="${r.sample_key}"></div><div>seed ${r.seed}${r.sample_key===best?' · FEM query':''}</div><div class="metric">T (${f(r.Ex_target)}, ${f(r.Ey_target)})<br>F (${f(r.Ex_FEM)}, ${f(r.Ey_FEM)})<br>ΔEx ${f(r.delta_Ex)}, ΔEy ${f(r.delta_Ey)}<br>d ${f(r.d_mech)} · ${esc(r.status)}</div><div class="buttons"><button onclick="view(this,'r')">旋转↺90°</button><button onclick="view(this,'f')">左右镜像</button><button onclick="view(this,'reset')">复位</button></div><div class="viewstate">原始显示</div><label>猫形 <select onchange="ratings['${r.sample_key}']=this.value"><option value="unreviewed">未审阅</option><option value="yes">可辨识</option><option value="uncertain">不确定</option><option value="no">不可辨识</option></select></label></div>`;}
function neighbor(r){return `<div class="card" data-index="${r.dataset_index}"><div class="picture"><img src="${r.display_image}" alt="${esc(r.sample_id)}"></div><div>#${r.rank} · ${esc(r.class_name)}<br>${esc(r.filename)}<br>ID ${esc(r.sample_id)}</div><div class="metric">original (${f(r.Ex_original)}, ${f(r.Ey_original)})<br>matched (${f(r.Ex_matched)}, ${f(r.Ey_matched)})<br>d ${f(r.distance)} · swap=${r.axis_swapped}</div><div>${esc(r.relative_path)}</div></div>`;}
for(const t of D.targets){let rr=D.rows.filter(r=>r.target_id===t.target_id),a=D.target_retrieval.find(r=>r.target_id===t.target_id),b=D.fem_retrieval.find(r=>r.target_id===t.target_id),s=D.summary.find(r=>r.target_id===t.target_id);let panel=document.createElement('section');panel.id=t.target_id;
panel.innerHTML=`<h2>${t.target_id} · s=${f(t.s_target)}, a=${f(t.a_target)} · Ex=${f(t.Ex_target)}, Ey=${f(t.Ey_target)}</h2><p>有效${s.n_valid}/${s.n_total}; mean(d)=${f(s.mean_d_mech)}; var(d)=${f(s.var_d_mech)}, ddof=1</p><h3>1. Generated（固定seeds顺序）</h3><div class="row">${rr.map(r=>generated(r,b.query_sample_key)).join('')}</div><h3>2. Target retrieval（按距离排名）</h3><p>Query: Ex=${f(a.query_Ex)}, Ey=${f(a.query_Ey)}; ${esc(D.distance_formula)}</p><div class="row">${a.neighbors.map(neighbor).join('')}</div><h3>3. FEM retrieval（按距离排名）</h3><p>Query来自 ${esc(b.query_sample_key||'无有效FEM')}，seed=${b.query_seed}; 真实FEM Ex=${f(b.query_Ex)}, Ey=${f(b.query_Ey)}; query对target误差=${f(b.query_target_distance)}; ${esc(D.distance_formula)}</p><div class="row">${b.neighbors.map(neighbor).join('')}</div><p>人工备注：猫形完整性、多样性、碎片化、可能记忆</p><textarea data-target="${t.target_id}" placeholder="手动备注；点击顶部导出JSON保存"></textarea>`;document.getElementById('panels').appendChild(panel);}
function downloadRatings(){const notes={};document.querySelectorAll('textarea[data-target]').forEach(e=>notes[e.dataset.target]=e.value);let blob=new Blob([JSON.stringify({run_id:D.run_id,ratings,notes,utc:new Date().toISOString()},null,2)],{type:'application/json'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=D.run_id+'_manual_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
</script></body></html>'''
