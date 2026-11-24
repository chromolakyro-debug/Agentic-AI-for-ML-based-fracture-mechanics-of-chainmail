import json,sys,time,platform,shutil,datetime,concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image
from utils import ROOT,HERE,read,save,sha,digest,pixel_hash,resolve,adapter,csvwrite,code_identity,to_sa,from_sa,distance,package_versions
from cache import get,put,valid_array
from retrieval import nearest,target_cache_key

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def summarize(rows,targets,ddof):
    summary=[]
    for t in targets:
        rr=[r for r in rows if r['target_id']==t['target_id']];ok=[r for r in rr if r['status']=='ok']
        d=np.array([r['d_mech'] for r in ok]);item=dict(target_id=t['target_id'],grid_row=t['grid_row'],grid_col=t['grid_col'],n_total=len(rr),n_valid=len(ok),n_failed=len(rr)-len(ok),
            mean_d_mech=float(d.mean()) if len(d) else None,var_d_mech=float(d.var(ddof=ddof)) if len(d)>ddof else None,
            joint_Ex_Ey_hit_rate=sum(r['hit_Ex_Ey'] for r in ok)/len(rr) if rr else None,
            joint_s_a_hit_rate=sum(r['hit_s_a'] for r in ok)/len(rr) if rr else None)
        for key in ('delta_Ex','delta_Ey','delta_s','delta_a'):
            v=np.array([r[key] for r in ok]);item.update({f'{key}_bias':float(v.mean()) if len(v) else None,
                f'{key}_abs_median':float(np.median(abs(v))) if len(v) else None,f'{key}_abs_p90':float(np.quantile(abs(v),.9)) if len(v) else None})
        summary.append(item)
    return summary

class Run:
    def __init__(self,config,root=ROOT,diagnostic=False):
        self.root=Path(root);self.cfg=read(config);self.diagnostic=diagnostic
        c=self.cfg;self.cache=resolve(c['cache'],self.root);self.cache.mkdir(parents=True,exist_ok=True)
        self.grid=read(resolve(c['grid'],self.root));self.seed_path=resolve(c['seeds'],self.root);self.seeds=read(self.seed_path)
        if len(self.seeds)!=10 or len(set(self.seeds))!=10:raise ValueError('Require exactly 10 fixed distinct seeds')
        if c['sampling']['batch_size']!=len(self.seeds):raise ValueError('This adapter batches all ten fixed seeds together')
        self.target_path=resolve(c['diagnostic_targets' if diagnostic else 'targets'],self.root);self.target_doc=read(self.target_path)
        self.targets=self.target_doc['targets']
        if not diagnostic:
            if self.target_doc['status']!='frozen':raise ValueError('BLOCKED: authoritative red-circle 24-cell selection missing. Use --diagnostic for independent module checks only.')
            from targets import make_targets
            expected=make_targets(self.target_doc['selected_cells'],self.grid)
            if self.targets!=expected or self.target_doc['grid_sha256']!=sha(resolve(c['grid'],self.root)):raise ValueError('Frozen target/grid mismatch')
        if not self.targets:raise ValueError('No targets')
        if len({t['target_id'] for t in self.targets})!=len(self.targets):raise ValueError('Duplicate target IDs')
        for t in self.targets:
            if not np.allclose(from_sa(t['s_target'],t['a_target'],self.grid['esoft']),[t['Ex_target'],t['Ey_target']],rtol=1e-13,atol=0):raise ValueError('Target conversion mismatch')
        if self.grid['esoft']!=c['fem']['parameters']['soft_modulus']:raise ValueError('Condition/FEM Esoft mismatch')
        self.dataset=adapter(c['dataset']['adapter'])(c['dataset'],self.root);audit=self.dataset.audit(self.cache)
        for key,value in c['model']['dataset_binding'].items():
            if self.dataset.files[key]!=value:raise ValueError(f'Model/dataset binding mismatch: {key}')
        if sha(resolve(c['model']['checkpoint'],self.root))!=c['model']['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        for field,key in [('E_soft','soft_modulus'),('E_hard','hard_modulus'),('nu_soft','poisson'),('nu_hard','poisson'),('specimen_width_mm','width'),('specimen_height_mm','height'),('specimen_thickness_mm','thickness'),('nominal_strain','strain')]:
            if any(float(r[field])!=c['fem']['parameters'][key] for r in self.dataset.rows):raise ValueError(f'Dataset/FEM mismatch: {field}')
        if not np.allclose(to_sa(self.dataset.moduli[:,0],self.dataset.moduli[:,1],self.grid['esoft']),self.dataset.data['sa'],atol=1e-14,rtol=1e-13):raise ValueError('Dataset label definition mismatch')
        self.out=(resolve(c['output_directory'],self.root) if c.get('output_directory') else HERE/'runs'/c['run_id'])
        if diagnostic:self.out=self.out.with_name(self.out.name+'_diagnostic')
        self.out.mkdir(parents=True,exist_ok=True)
        self.distance=c['retrieval']['distance'];self.k=c['retrieval']['top_k']
        identity=dict(config=c,dataset_identity=self.dataset.identity,training_identity=self.dataset.training_identity,targets=self.target_doc,seeds=self.seeds,grid=self.grid)
        self.identity=digest(identity);manifest_path=self.out/'manifest.json'
        if manifest_path.exists() and read(manifest_path)['identity']!=self.identity:raise ValueError('Run inputs changed: choose a new run_id; preserve old results')
        if not manifest_path.exists():save(manifest_path,dict(identity=self.identity,created_utc=now(),diagnostic_only=diagnostic,inputs=identity,dataset_audit=audit,
            checkpoint=c['model']['checkpoint'],checkpoint_sha256=c['model']['checkpoint_sha256'],platform=platform.platform(),python=sys.version,
            note='Retrieval uses axis swap even though legacy training augmentation was none. No new training.'))
        self.codes=code_identity();save(self.out/'code_hashes.json',self.codes)
        manifest=read(manifest_path)
        history=[]
        if (self.out/'events.jsonl').exists():
            for line in (self.out/'events.jsonl').read_text(encoding='utf-8').splitlines():
                event=json.loads(line)
                if event['stage']=='invocation':history.append(dict(utc=event['utc'],code_sha256=event['code_sha256']))
        manifest.update(code_versions=history+[dict(utc=now(),code_sha256=self.codes)],
            library_versions=package_versions(['numpy','scipy','pyamg','torch','Pillow','matplotlib','threadpoolctl']))
        save(manifest_path,manifest)
        snapshots=self.out/'code_snapshot'
        for rel,h in self.codes.items():
            dst=snapshots/h/Path(rel).name;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copy2(self.root/rel,dst)
        save(self.out/'config.json',c);save(self.out/'targets.json',self.target_doc);save(self.out/'seeds.json',self.seeds);save(self.out/'grid.json',self.grid)
        self.fem_adapter=adapter(c['fem']['adapter']);self.fem_identity=self.fem_adapter.identity(c['fem'],self.root)
        self.events=[]
    def event(self,stage,**details):
        item=dict(utc=now(),stage=stage,**details);self.events.append(item)
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        print(json.dumps(item,ensure_ascii=False),flush=True)
    def neighbors(self,query):
        result=nearest(query,self.dataset.moduli,self.dataset.source_keys,self.dataset.train,self.distance,self.k)
        return [{**self.dataset.describe(r['dataset_index']),**r} for r in result]
    def prepare_retrieval(self):
        implementation=digest({name:sha(HERE/name) for name in ['retrieval.py','utils.py','adapters/dataset.py']})
        key=target_cache_key(self.dataset.training_identity,self.targets,self.distance,self.k,implementation)
        path=self.cache/'target_retrieval'/f'{key}.json';result=get(path,key);hit=result is not None
        if result is None:
            result=[dict(target_id=t['target_id'],query_Ex=t['Ex_target'],query_Ey=t['Ey_target'],neighbors=self.neighbors([t['Ex_target'],t['Ey_target']])) for t in self.targets]
            put(path,key,result)
        save(self.out/'target_retrieval.json',result);csvwrite(self.out/'target_retrieval.csv',[{**{k:v for k,v in q.items() if k!='neighbors'},**r} for q in result for r in q['neighbors']])
        self.event('target_retrieval',cache_hit=hit,cache_key=key,targets=len(result),training=len(self.dataset.train))
    def generate(self):
        start=time.perf_counter();model=adapter(self.cfg['model']['adapter'])(self.cfg['model'],self.cfg['sampling'],self.dataset,self.root)
        save(self.out/'model_info.json',model.info)
        noise=model.initial_noise(self.seeds);np.save(self.out/'initial_noise.npy',noise)
        save(self.out/'initial_noise.json',dict(seeds=self.seeds,sha256=sha(self.out/'initial_noise.npy'),per_seed_pixel_sha256=[pixel_hash(n) for n in noise],shape=list(noise.shape)))
        prep=self.cfg['preprocessing']
        assert prep['comparison']=='>=' and prep['black_hard']==0 and prep['white_soft']==1 and not prep['resize'] and not prep['flip']
        records=[];cached=0
        for t in self.targets:
            key=digest(dict(model=self.cfg['model'],sampling=self.cfg['sampling'],preprocessing=prep,target=t,seeds=self.seeds,model_info=model.info,
                code={f:self.codes[f] for f in ['cfm_v0/model.py','evaluation/adapters/cfm.py','evaluation/pipeline.py','evaluation/utils.py']},noise_sha256=sha(self.out/'initial_noise.npy')))
            path=self.out/'arrays'/f'{t["target_id"]}.npz';meta_path=path.with_suffix('.json');meta=valid_array(path,meta_path,key);was_cached=meta is not None
            if meta is None:
                tick=time.perf_counter();raw,condition=model.generate(t,self.seeds);binary=np.ascontiguousarray(raw>=prep['threshold'],dtype=np.uint8)
                assert raw.shape==binary.shape==(len(self.seeds),*self.cfg['sampling']['shape'])
                atomic_npz(path,raw=raw,binary=binary,seeds=np.array(self.seeds,dtype=np.int64),standardized_condition=condition)
                meta=dict(file_sha256=sha(path),seconds=time.perf_counter()-tick,binary_pixel_sha256=[pixel_hash(b) for b in binary]);put(meta_path,key,meta)
            else:cached+=1
            with np.load(path,allow_pickle=False) as f:binary=f['binary'];condition=f['standardized_condition']
            for j,seed in enumerate(self.seeds):
                name=f'{t["target_id"]}_{seed}';png=self.out/'images'/f'{name}.png';png.parent.mkdir(exist_ok=True)
                Image.fromarray(binary[j]*255,mode='L').save(png)
                assert np.array_equal(np.asarray(Image.open(png))//255,binary[j])
                records.append(dict(sample_key=name,**t,seed=seed,checkpoint=self.cfg['model']['checkpoint'],checkpoint_sha256=self.cfg['model']['checkpoint_sha256'],
                    generation_key=key,raw_array=path.relative_to(self.out).as_posix(),array_index=j,image=png.relative_to(self.out).as_posix(),png_sha256=sha(png),
                    binary_pixel_sha256=pixel_hash(binary[j]),condition_s_standardized=float(condition[0]),condition_a_standardized=float(condition[1])))
            self.event('generate_target',target=t['target_id'],cache_hit=was_cached,original_sampling_seconds=meta['seconds'])
        save(self.out/'generated.json',records);csvwrite(self.out/'generated.csv',records)
        self.event('generate',images=len(records),cached_targets=cached,seconds=time.perf_counter()-start)
    def generated(self):
        records=read(self.out/'generated.json');arrays={}
        for r in records:
            path=self.out/r['raw_array']
            if path not in arrays:
                meta=valid_array(path,path.with_suffix('.json'),r['generation_key'])
                if not meta:raise ValueError(f'Generation archive changed: {path}')
                arrays[path]=np.load(path,allow_pickle=False)['binary']
            b=arrays[path][r['array_index']]
            if pixel_hash(b)!=r['binary_pixel_sha256'] or sha(self.out/r['image'])!=r['png_sha256']:raise ValueError('Generation image checksum mismatch')
            if not np.array_equal(np.asarray(Image.open(self.out/r['image']))//255,b):raise ValueError('HTML image differs from FEM material')
            yield r,b
    def fem_key(self,b):return digest(dict(fem=self.fem_identity,shape=list(b.shape),dtype=str(b.dtype),pixels=pixel_hash(b)))
    def fem(self,retry_failed=False):
        start=time.perf_counter();items=list(self.generated());pending={};results={};hits=0
        for r,b in items:
            key=self.fem_key(b);p=self.cache/'fem'/f'{key}.json';v=get(p,key)
            if v is not None and not (v['status']=='failed' and retry_failed):results[key]=v;hits+=1
            else:
                if v is not None:shutil.copy2(p,p.with_suffix(f'.failed.{time.time_ns()}.json'))
                pending[key]=b
        if pending:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.cfg['fem']['workers'],initializer=self.fem_adapter.initializer,initargs=(self.cfg['fem'],)) as pool:
                futures={pool.submit(self.fem_adapter.solve,(key,b)):key for key,b in pending.items()}
                for i,f in enumerate(concurrent.futures.as_completed(futures),1):
                    key=futures[f]
                    try:v=f.result()
                    except Exception as e:v=dict(status='failed',fem_valid=False,error=repr(e),cache_key=key,input_pixel_sha256=pixel_hash(pending[key]))
                    if v['input_pixel_sha256']!=pixel_hash(pending[key]):raise ValueError('FEM result/input mismatch')
                    put(self.cache/'fem'/f'{key}.json',key,v);results[key]=v
                    if i==1 or i%10==0 or i==len(futures):self.event('fem_progress',completed=i,total=len(futures),seconds=time.perf_counter()-start)
        rows=[]
        for r,b in items:
            key=self.fem_key(b);v=results[key];row={**r,**v,'fem_config_identity':self.fem_identity}
            if v['status']=='ok':
                ex,ey=v['Ex_app'],v['Ey_app'];s,a=to_sa(ex,ey,self.grid['esoft']);tol=self.cfg['tolerances']
                row.update(Ex_FEM=ex,Ey_FEM=ey,s_FEM=float(s),a_FEM=float(a),delta_Ex=ex-r['Ex_target'],delta_Ey=ey-r['Ey_target'],delta_s=float(s-r['s_target']),delta_a=float(a-r['a_target']),
                    d_mech=float(distance([ex,ey],[r['Ex_target'],r['Ey_target']],self.distance)))
                for k in ['Ex','Ey']:
                    row[f'abs_error_{k}']=abs(row[f'delta_{k}']);row[f'relative_error_{k}']=abs(row[f'delta_{k}'])/max(abs(r[f'{k}_target']),tol['relative_denominator_floor'])
                row['hit_Ex_Ey']=row['relative_error_Ex']<=tol['Ex_relative'] and row['relative_error_Ey']<=tol['Ey_relative']
                row['hit_s_a']=abs(row['delta_s'])<=tol['s'] and abs(row['delta_a'])<=tol['a']
            rows.append(row)
        save(self.out/'results.json',rows);csvwrite(self.out/'results.csv',rows)
        self.event('fem',images=len(rows),solved_unique=len(pending),cache_hits=hits,failed=sum(r['status']!='ok' for r in rows),seconds=time.perf_counter()-start)
    def fem_retrieval(self):
        rows=read(self.out/'results.json');result=[]
        for t in self.targets:
            valid=[r for r in rows if r['target_id']==t['target_id'] and r['status']=='ok']
            if not valid:result.append(dict(target_id=t['target_id'],status='no_valid_FEM',neighbors=[]));continue
            best=min(valid,key=lambda r:(r['d_mech'],self.seeds.index(r['seed'])))
            result.append(dict(target_id=t['target_id'],status='ok',query_sample_key=best['sample_key'],query_seed=best['seed'],query_Ex=best['Ex_FEM'],query_Ey=best['Ey_FEM'],
                query_target_distance=best['d_mech'],selection='minimum d_mech among valid generated seeds; FEM-selected best-of-10',neighbors=self.neighbors([best['Ex_FEM'],best['Ey_FEM']])))
        save(self.out/'fem_retrieval.json',result);csvwrite(self.out/'fem_retrieval.csv',[{**{k:v for k,v in q.items() if k!='neighbors'},**r} for q in result for r in q['neighbors']])
        self.event('fem_retrieval',queries=len(result))
    def statistics(self):
        rows=read(self.out/'results.json');summary=summarize(rows,self.targets,self.cfg['statistics']['variance_ddof'])
        save(self.out/'summary.json',summary);csvwrite(self.out/'summary.csv',summary);self.event('statistics',targets=len(summary))
    def report(self):
        from reporting import report
        report(self);self.event('report',path=str(self.out/'index.html'))

def execute(args):
    start=time.perf_counter();run=Run(args.config,args.root,args.diagnostic)
    run.event('invocation',argv=sys.argv,code_sha256=run.codes)
    stages=['target_retrieval','generate','fem','fem_retrieval','statistics','report'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='audit':run.event('audit',result=run.dataset.audit(run.cache))
        elif stage=='target_retrieval':run.prepare_retrieval()
        elif stage=='fem':run.fem(args.retry_failed)
        else:getattr(run,stage)()
    run.event('complete',seconds=time.perf_counter()-start,stages=stages)
