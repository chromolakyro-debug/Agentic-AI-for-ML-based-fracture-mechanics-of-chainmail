import csv
from collections import Counter
import numpy as np
from PIL import Image
from utils import resolve,read,sha,digest,pixel_hash,save
from cache import get,put

class NumpyMetadataDataset:
    """Explicit ID-to-row join plus pixel/FEM-label checks, old and curated releases."""
    def __init__(self,cfg,root):
        self.cfg=cfg;self.root=root;self.base=resolve(cfg['directory'],root)
        with (self.base/cfg['metadata']).open(encoding='utf-8-sig',newline='') as f:self.rows=list(csv.DictReader(f))
        self.images=np.load(self.base/cfg['array'],mmap_mode='r',allow_pickle=False)
        self.data=np.load(resolve(cfg['labels'],root),allow_pickle=False)
        self.splits=read(resolve(cfg['splits'],root))
        self.train=np.array(self.splits[cfg['training_index_key']],dtype=int)
        self.moduli=np.array([[float(r[cfg['Ex_field']]),float(r[cfg['Ey_field']])] for r in self.rows])
        assert len(self.rows)==len(self.images) and self.images.dtype==np.uint8
        assert np.array_equal(self.moduli,self.data['moduli'])
        assert len(set(self.train))==len(self.train)
        seen=set()
        for key in ('train','validation','test'):
            ids=self.splits[key];assert not seen&set(ids);seen.update(ids)
            assert np.array_equal(ids,self.data[key])
        assert seen==set(range(len(self.rows)))
        self.ids=[r[cfg['id_field']] for r in self.rows];assert len(set(self.ids))==len(self.ids)
        if cfg.get('npz_id_key'):assert np.array_equal(self.ids,self.data[cfg['npz_id_key']])
        self.paths=[self.base/cfg['image_prefix']/r[cfg['image_field']] for r in self.rows]
        self.source_keys=[]
        for r in self.rows:
            for fields in cfg['dedup_field_priority']:
                if all(r.get(k) for k in fields):
                    self.source_keys.append(digest({k:r[k] for k in fields}));break
            else:raise ValueError('Missing source identity; configure dedup fields explicitly')
        # Content hashes, not filename/mtime, determine cache identity.
        self.png_hashes=[sha(p) for p in self.paths]
        files={'metadata':sha(self.base/cfg['metadata']),'array':sha(self.base/cfg['array']),
            'labels':sha(resolve(cfg['labels'],root)),'splits':sha(resolve(cfg['splits'],root))}
        self.identity=digest(dict(config=cfg,files=files,png_hashes=self.png_hashes))
        self.files=files
        self.training_identity=digest(dict(dataset=self.identity,indices=self.train.tolist(),sources=[self.source_keys[i] for i in self.train]))
    def audit(self,cache):
        audit_identity=digest(dict(dataset=self.identity,adapter=sha(__file__)))
        path=cache/'dataset'/f'{audit_identity}.json';cached=get(path,audit_identity)
        if cached is not None:return cached
        for i,(r,p) in enumerate(zip(self.rows,self.paths)):
            a=np.asarray(Image.open(p).convert('L'));assert np.isin(a,[0,255]).all()
            b=(a//255).astype(np.uint8);assert np.array_equal(b,self.images[i])
            assert pixel_hash(b)==r[self.cfg['pixel_hash_field']]
            if self.cfg.get('id_numeric_prefix'):
                # Old labels have no explicit IDs; validate established numeric ID against every array row.
                assert self.ids[i]==f'{self.cfg["id_numeric_prefix"]}{i:04d}.png'
        result=dict(status='passed',samples=len(self.rows),training=len(self.train),
            unique_training_sources=len({self.source_keys[i] for i in self.train}),identity=self.identity,files=self.files)
        put(path,audit_identity,result);return result
    def describe(self,index):
        r=self.rows[index]
        return dict(dataset_index=int(index),sample_id=self.ids[index],source_key=self.source_keys[index],
            class_name=r[self.cfg['class_field']],filename=r[self.cfg['image_field']],
            relative_path=self.paths[index].relative_to(self.base).as_posix(),
            Ex_original=float(self.moduli[index,0]),Ey_original=float(self.moduli[index,1]),
            png_sha256=self.png_hashes[index],pixel_sha256=r[self.cfg['pixel_hash_field']])
