import importlib.util
import numpy as np
import torch
from utils import resolve,sha

class CFMAdapter:
    """Reuses the existing U-Net and midpoint sampler, with explicit EMA/config binding."""
    def __init__(self,cfg,sampling,dataset,root):
        self.cfg=cfg;self.sampling=sampling
        for key,value in cfg['dataset_binding'].items():
            if dataset.files[key]!=value:raise ValueError(f'Checkpoint/dataset mismatch: {key}')
        if cfg['extra_conditioning']:raise ValueError('This legacy CFM has only s,a conditioning')
        self.mean=np.array(cfg['normalization']['mean']);self.std=np.array(cfg['normalization']['std'])
        assert np.array_equal(self.mean,dataset.data['mean']) and np.array_equal(self.std,dataset.data['std'])
        assert sampling['method']=='midpoint' and sampling['shape']==[224,224]
        assert sampling['precision']=='bfloat16' and sampling['device']=='cuda'
        assert sampling['noise_generator']=='torch_cuda_generator_per_seed'
        spec=importlib.util.spec_from_file_location('evaluation_reused_cfm',resolve(cfg['module_file'],root));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.set_num_threads(sampling['cpu_threads']);torch.manual_seed(sampling['global_seed'])
        torch.backends.cudnn.benchmark=sampling['cudnn_benchmark']
        torch.backends.cuda.matmul.allow_tf32=sampling['allow_tf32'];torch.backends.cudnn.allow_tf32=sampling['allow_tf32']
        torch.backends.cudnn.deterministic=sampling['cudnn_deterministic']
        self.checkpoint=resolve(cfg['checkpoint'],root);self.checkpoint_sha256=sha(self.checkpoint)
        if self.checkpoint_sha256!=cfg['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        checkpoint=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        self.model=getattr(module,cfg['class_name'])(**cfg['architecture']).to(sampling['device']).to(memory_format=torch.channels_last)
        self.model.load_state_dict(checkpoint[cfg['weights_key']],strict=True);self.model.eval()
        self.sample=module.sample;self.step=int(checkpoint['step'])
        self.info=dict(step=self.step,parameters=sum(p.numel() for p in self.model.parameters()),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            checkpoint_sha256=self.checkpoint_sha256,training_augmentation=checkpoint['config'].get('augmentation','unspecified'))
    def condition(self,target):return (np.array([target['s_target'],target['a_target']])-self.mean)/self.std
    def initial_noise(self,seeds):
        return torch.stack([torch.randn((1,*self.sampling['shape']),generator=torch.Generator(device=self.sampling['device']).manual_seed(int(seed)),device=self.sampling['device']) for seed in seeds]).cpu().numpy()
    def generate(self,target,seeds):
        condition=self.condition(target)
        raw=self.sample(self.model,condition,seeds,steps=self.sampling['steps'],device=self.sampling['device']).float().cpu().numpy()
        assert np.isfinite(raw).all();return raw,condition
