"""Read independent FEM results, verify frozen inputs, report failures without filtering."""
from collections import Counter
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run import ROOT,DATA,OUT,HERE,read,save,sha,csvsave,transform,map_ex_ey,pixels_sha


def records(mode):
    return {r['key']:r for line in (OUT/(mode+'_fem.jsonl')).read_text().splitlines() if line.strip() for r in [json.loads(line)]}


def distribution(values):
    return dict(median=float(np.median(values)),p90=float(np.percentile(values,90)),maximum=float(np.max(values))) if values else None


def main():
    cfg=read(OUT/'config.json');samples=read(OUT/'samples.json');raw=records('main')
    assert len(raw)==320
    for p,digest in read(OUT/'frozen_inputs.json').items():assert sha(OUT/p)==digest
    for p,digest in cfg['protected_files_sha256'].items():assert sha(ROOT/p)==digest,('Source changed',p)
    images=np.load(DATA/'dataset.npy',mmap_mode='r',allow_pickle=False)
    output=[]
    for s in samples:
        n=s['sample_order'];orig=raw[f'{n:03d}_original'];aug=raw[f'{n:03d}_augmented']
        assert orig['pixel_sha256']==s['original_pixel_sha256'] and aug['pixel_sha256']==s['augmented_pixel_sha256']
        assert orig['config_sha256']==aug['config_sha256']==sha(OUT/'config.json')
        r=dict(s,original_status=orig['status'],augmented_status=aug['status'],
            Ex_original_FEM=None,Ey_original_FEM=None,Ex_expected=None,Ey_expected=None,
            Ex_augmented_FEM=None,Ey_augmented_FEM=None,Ex_abs_error=None,Ey_abs_error=None,Ex_rel_error=None,Ey_rel_error=None,
            Ex_metadata_abs_error=None,Ey_metadata_abs_error=None,Ex_metadata_rel_error=None,Ey_metadata_rel_error=None,
            Ex_pass=False,Ey_pass=False,metadata_pass=False,mapping_pass=False,overall_pass=False)
        if orig['fem_valid']:
            original=np.array([orig['Ex_app'],orig['Ey_app']]);meta=np.array([s['Ex_metadata'],s['Ey_metadata']])
            expected=np.array(map_ex_ey(*original,s['quarter_turns']))
            meta_error=np.abs(original-meta)
            r['metadata_pass']=bool(np.all(meta_error<=cfg['atol']+cfg['rtol']*np.abs(meta)))
            for j,axis in enumerate(('x','y')):
                r[f'E{axis}_original_FEM']=float(original[j]);r[f'E{axis}_expected']=float(expected[j])
                r[f'E{axis}_metadata_abs_error']=float(meta_error[j]);r[f'E{axis}_metadata_rel_error']=float(meta_error[j]/max(abs(meta[j]),cfg['relative_denominator_floor']))
            if aug['fem_valid']:
                actual=np.array([aug['Ex_app'],aug['Ey_app']]);err=np.abs(actual-expected)
                for j,axis in enumerate(('x','y')):
                    r[f'E{axis}_augmented_FEM']=float(actual[j]);r[f'E{axis}_abs_error']=float(err[j])
                    r[f'E{axis}_rel_error']=float(err[j]/max(abs(expected[j]),cfg['relative_denominator_floor']))
                    r[f'E{axis}_pass']=bool(err[j]<=cfg['atol']+cfg['rtol']*abs(expected[j]))
                r['mapping_pass']=r['Ex_pass'] and r['Ey_pass']
        r['overall_pass']=r['mapping_pass'] and r['metadata_pass']
        for kind,result in [('original',orig),('augmented',aug)]:
            for key in ['solve_seconds','residual_x','residual_y','iterations_x','iterations_y','energy_relative_error_x','energy_relative_error_y','force_balance_relative_error_x','force_balance_relative_error_y']:
                r[kind+'_'+key]=result.get(key)
            r[kind+'_error']=result.get('error','')
        output.append(r)
    csvsave(OUT/'per_sample_results.csv',output);save(OUT/'per_sample_results.json',output)
    failures=[r for r in output if not r['overall_pass']];save(OUT/'failures.json',failures)
    summary=dict(n=len(output),fem_solves=320,fem_failures=sum(not r['fem_valid'] for r in raw.values()),
        mapping_pass=sum(r['mapping_pass'] for r in output),metadata_pass=sum(r['metadata_pass'] for r in output),
        overall_pass=sum(r['overall_pass'] for r in output),protected_files_unchanged=len(cfg['protected_files_sha256']),
        atol=cfg['atol'],rtol=cfg['rtol'],source_and_splits_unchanged=True,
        error_distribution={key:distribution([r[key] for r in output if r[key] is not None]) for key in
            ['Ex_abs_error','Ey_abs_error','Ex_rel_error','Ey_rel_error','Ex_metadata_rel_error','Ey_metadata_rel_error']},
        solver_diagnostics={key:distribution([r[key] for r in raw.values() if r.get(key) is not None]) for key in
            ['residual_x','residual_y','energy_relative_error_x','energy_relative_error_y','force_balance_relative_error_x','force_balance_relative_error_y']})
    def group(rows,label):
        errs=[r[key] for r in rows for key in ('Ex_rel_error','Ey_rel_error') if r[key] is not None]
        return dict(label=label,n=len(rows),passed=sum(r['mapping_pass'] for r in rows),error=distribution(errs))
    summary['by_transform']=[group([r for r in output if r['angle_ccw']==angle and r['horizontal_flip']==flip],f'{angle}deg_flip{int(flip)}') for angle in (0,90,180,270) for flip in (False,True)]
    summary['by_class']=[group([r for r in output if r['group_id']==g],g) for g in sorted({r['group_id'] for r in output})]
    summary['by_swap']=[group([r for r in output if r['swap_ex_ey']==swap],str(swap)) for swap in (False,True)]
    precision=[]
    for mode in ('precision','direct'):
        checks=records(mode);orders=cfg['precision_sample_orders' if mode=='precision' else 'direct_sample_orders']
        assert len(checks)==2*len(orders)
        for n in orders:
            orig=checks[f'{n:03d}_original'];aug=checks[f'{n:03d}_augmented'];s=samples[n]
            r=dict(mode=mode,sample_order=n,new_id=s['new_id'],valid=orig['fem_valid'] and aug['fem_valid'],
                max_mapping_relative_error=None,max_vs_main_relative_error=None,mapping_pass=False,vs_main_pass=False)
            if r['valid']:
                expected=np.array(map_ex_ey(orig['Ex_app'],orig['Ey_app'],s['quarter_turns']))
                actual=np.array([aug['Ex_app'],aug['Ey_app']]);error=np.abs(actual-expected)
                r['max_mapping_relative_error']=float(np.max(error/np.abs(expected)))
                r['mapping_pass']=bool(np.all(error<=cfg['atol']+cfg['rtol']*np.abs(expected)))
                discrepancies=[];passes=[]
                for kind in ('original','augmented'):
                    high=checks[f'{n:03d}_{kind}'];base=raw[f'{n:03d}_{kind}']
                    assert high['pixel_sha256']==base['pixel_sha256'] and high['config_sha256']==base['config_sha256']
                    for axis in ('x','y'):
                        a=high[f'E{axis}_app'];b=base[f'E{axis}_app'];discrepancies.append(abs(b-a)/abs(a));passes.append(abs(b-a)<=cfg['atol']+cfg['rtol']*abs(a))
                r['max_vs_main_relative_error']=max(discrepancies);r['vs_main_pass']=all(passes)
            precision.append(r)
    csvsave(OUT/'precision_comparison.csv',precision);save(OUT/'precision_comparison.json',precision)
    summary['precision']=dict(pairs=len(precision),valid=sum(r['valid'] for r in precision),
        mapping_pass=sum(r['mapping_pass'] for r in precision),vs_main_pass=sum(r['vs_main_pass'] for r in precision),
        max_mapping_relative_error=max(r['max_mapping_relative_error'] for r in precision if r['valid']),
        max_vs_main_relative_error=max(r['max_vs_main_relative_error'] for r in precision if r['valid']))
    summary['timing']={mode:read(OUT/(mode+'_timing.json')) for mode in ('main','precision','direct')}
    summary['complete_pass']=not failures and all(r['mapping_pass'] and r['vs_main_pass'] for r in precision)
    save(OUT/'summary.json',summary)
    # A single deterministic, outcome-independent contact sheet, native224px tiles.
    canvas=Image.new('RGB',(1940,1500),'#f3f5f7');draw=ImageDraw.Draw(canvas)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    draw.text((22,12),'精确旋转 + 左右翻转：原图与实际 FEM 验证',font=font(30),fill='#183748')
    draw.text((22,57),'每类首个随机样本（求解前固定）；先逆时针旋转，后 Horizontal Flip；像素 0=硬，1=软',font=font(20),fill='#50616b')
    for j,n in enumerate(cfg['figure_sample_orders']):
        r=output[n];x=20+(j%2)*970;y=103+(j//2)*343
        draw.text((x,y),f'{r["group_id"]} {r["group_name_zh"]} | {r["new_id"]} | {r["angle_ccw"]}° CCW + '+('左右翻转' if r['horizontal_flip'] else '不翻转'),font=font(21),fill='#183748')
        arr=images[r['dataset_index']];aug=transform(arr,r['quarter_turns'],r['horizontal_flip'])
        for k,(name,pix) in enumerate([('原图',arr),('增强图',aug)]):
            xx=x+k*236;yy=y+68
            draw.text((xx,y+36),name,font=font(18),fill='#50616b')
            canvas.paste(Image.fromarray(pix*255).convert('RGB'),(xx,yy))
            draw.rectangle((xx-1,yy-1,xx+224,yy+224),outline='#ccd2d7')
        xx=x+484
        lines=['Ex / Ey（固定全局坐标）',
            f'原图 FEM: {r["Ex_original_FEM"]:.7f} / {r["Ey_original_FEM"]:.7f}',
            f'预期:       {r["Ex_expected"]:.7f} / {r["Ey_expected"]:.7f}',
            f'增强 FEM: {r["Ex_augmented_FEM"]:.7f} / {r["Ey_augmented_FEM"]:.7f}',
            f'相对误差: {r["Ex_rel_error"]:.2e} / {r["Ey_rel_error"]:.2e}',
            '映射：'+('交换 Ex / Ey' if r['swap_ex_ey'] else 'Ex / Ey 不变'),
            '判定：'+('通过' if r['overall_pass'] else '失败，见逐样本记录')]
        for l,text in enumerate(lines):draw.text((xx,y+64+l*31),text,font=font(18),fill='#183748')
    draw.text((22,1470),'展示样本未按误差筛选。完整统计覆盖160张原图及160次增强图独立FEM；本轮没有训练或写入增强数据集。',font=font(17),fill='#50616b')
    canvas.save(OUT/'augmentation_comparison.png')
    report(summary,cfg,output,precision)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def report(summary,cfg,rows,precision):
    d=summary['error_distribution'];times=summary['timing'];elapsed=sum(x['invocation_seconds'] for x in times.values())
    text=['# 旋转与镜像 augmentation 力学标签验证','',
        f'结论：160组中{summary["mapping_pass"]}/160组满足Ex/Ey映射的预定混合判据；原图重算与metadata {summary["metadata_pass"]}/160组一致。主验证320次图像FEM（每次含x/y两种加载），求解失败{summary["fem_failures"]}次。固定精度复核另做20次图像FEM。'+('当前Ex/Ey标签可按本报告规则用于on-the-fly augmentation。' if summary['complete_pass'] else '存在未通过项，暂不作无条件可用结论。'),'',
        '每类无放回随机20张，共160张；seed=2026111601，抽样和变换使用两个独立PCG64随机流。抽样覆盖现有全部split，仅做验证，不迁移集合。每张均匀抽取k∈{0,1,2,3}，独立抽取50% flip，先np.rot90(k)，再np.flip(axis=1)。未为凑齐变换重新抽样；恒等变换也实际重复求解。样本及变换在求解前冻结于samples.json/config.json。', '',
        '图像列随物理x增大，图像行向下；mechanics.py用flipud将材料赋到y向上的网格。np.rot90的正角为物理逆时针，axis=1 flip为物理左右镜像。输入输出都是224×224 uint8、严格0/1、连续内存，未反色、插值或重编码源PNG；每对相位像素数和体积分数严格一致。', '',
        '适用物理前提：40×40×2mm正方形试件，hx=hy=40/224mm；每像素一个全积分Q4 plane-stress单元，两相各向同性linear elasticity，Esoft=1、Ehard=10、相同nu=0.3。x拉伸左右边规定ux=0/0.04mm、横向自由，上下traction-free；y拉伸交换方向。Ex=Fx/(Ly·thickness·strain)，Ey=Fy/(Lx·thickness·strain)，都在固定全局坐标定义。旋转/镜像只是网格与材料的置换；没有周期边界配对。原点单个横向pin只消除刚体平移，换角点不改变应变能或模量；反向加载经刚体平移/符号变换等价。未发现破坏该映射的方向性材料或非对称变形约束，未改原力学问题。', '',
        '原图与增强图均从各自像素重新装配刚度矩阵并实际求解，没有用交换输出代替增强FEM。生产设置为float64、AMG-CG、tol=1e-9、maxiter=1200；复用网格/预条件器，不复用材料矩阵或解。', '',
        f'求解前固定：abs(actual−expected)≤1e-8+1e-7·abs(expected)，atol以当前Esoft=1的模量单位计；相对误差分母max(abs(expected),1e-12·Esoft)。当前两相实心且同nu，模量能量界为[1,10]，尺度下限不实际触发。依据为既有AMG/direct回归阈值1e-7，以及既有224分辨率对照误差{cfg["historical_direct_relative_discrepancy"]:.2e}；残差容差本身不等于模量误差。本轮每类预先固定1对样本以tol=1e-11复核，并对01、05类各1对做独立spsolve直接解。没有事后放宽容差。','',
        '| CCW旋转 | 左右翻转 | 样本数 | 映射通过 | 最大相对误差（Ex/Ey合并） |','|---|---|---:|---:|---:|']
    for g in summary['by_transform']:
        angle,flip=g['label'].split('deg_flip');text.append(f'| {angle}° | {"是" if flip=="1" else "否"} | {g["n"]} | {g["passed"]} | {g["error"]["maximum"]:.3e} |')
    text+=['','| 量 | 绝对误差 median / P90 / max | 相对误差 median / P90 / max |','|---|---|---|']
    for axis in ('x','y'):
        aa=d[f'E{axis}_abs_error'];rr=d[f'E{axis}_rel_error']
        text.append(f'| E{axis} | '+ ' / '.join(f'{aa[k]:.3e}' for k in ('median','p90','maximum'))+' | '+' / '.join(f'{rr[k]:.3e}' for k in ('median','p90','maximum'))+' |')
    text+=['',f'原图重算对metadata最大相对误差：Ex {d["Ex_metadata_rel_error"]["maximum"]:.3e}、Ey {d["Ey_metadata_rel_error"]["maximum"]:.3e}。按类统计见summary.json；失败样本完整保存在failures.json（当前{160-summary["overall_pass"]}条），全部原始求解诊断在三个*_fem.jsonl中。', '',
        f'精度复核：10对中{summary["precision"]["mapping_pass"]}对映射通过，生产解对高精度解最大相对变化{summary["precision"]["max_vs_main_relative_error"]:.3e}；高精度映射最大相对误差{summary["precision"]["max_mapping_relative_error"]:.3e}。这支持误差来自迭代/浮点计算而非标签映射错误。此次没有超差后修复或重跑，也未细化网格：所验映射是同一正方形Q4网格的对称置换，不是对连续体绝对模量离散精度的证明。', '',
        f'0°/180°和任意左右flip下，Ex/Ey保持；90°/270°和任意左右flip下，Ex/Ey交换，均以本轮抽样和预定容差为限。可在training样本读取时调用augmentation.py的random_augmentation，先对0/1图及未标准化Ex/Ey变换，再执行原有条件转换和标准化；不直接置换已标准化条件。本次只验证Ex/Ey，不默认Poisson ratio、剪切量、方向性场、姿态metadata等也遵循同一规则；未修改CFM训练代码或启动训练。', '',
        '局限：160张抽样并非全量7857张穷举；使用全部split进行的本次物理不变性验证不构成训练或模型选择。该结论限于当前正方形、等间距、各向同性、对称加载协议，不能直接推广至矩形区域、方向性材料、其他边界条件或任意角度旋转。没有发现当前Ex/Ey实现的未解决问题。', '',
        f'主验证实测wall time {times["main"]["invocation_seconds"]:.1f}s（8 CPU workers），单图含两向加载的中位求解时间{times["main"]["median_solve_seconds"]:.2f}s；高精度迭代{times["precision"]["invocation_seconds"]:.1f}s、直接解{times["direct"]["invocation_seconds"]:.1f}s。三阶段wall time之和{elapsed:.1f}s；后两阶段并行，此和不是端到端用时。所有源数据、metadata、split和现有solver/loader文件hash未变。只保留变换配方和一张对照图，没有永久增强训练集副本。', '',
        '复现（项目根目录、现有环境）：','', '```powershell',
        '.\\cfm_v0\\.venv\\Scripts\\python.exe augmentation_validation\\run.py prepare  # 首次冻结；已有结果时禁止重复prepare',
        '.\\cfm_v0\\.venv\\Scripts\\python.exe augmentation_validation\\run.py main',
        '.\\cfm_v0\\.venv\\Scripts\\python.exe augmentation_validation\\run.py precision',
        '.\\cfm_v0\\.venv\\Scripts\\python.exe augmentation_validation\\run.py direct',
        '.\\cfm_v0\\.venv\\Scripts\\python.exe augmentation_validation\\analyze.py','```','',
        'main/precision/direct支持按配置hash、像素hash与原图/增强图独立key断点续算；已有失败记录不会被静默覆盖。要独立从零重跑，应复制脚本到另一个项目同级实验目录，保留本次冻结结果。']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':main()
