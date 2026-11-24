"""Update-based cosine restart and full-cycle, two-stage plateau controller."""
import math
class RestartSchedule:
    def __init__(self,optimizer,cfg):self.optimizer=optimizer;self.cfg=cfg;self.updates=0
    def value(self,update):
        # update is one-based. A complete cycle contains exactly cycle_steps optimizer updates.
        i=(update-1)//self.cfg['cycle_steps'];position=(update-1)%self.cfg['cycle_steps']
        peak=max(self.cfg['peak_floor'],self.cfg['max_lr']*self.cfg['peak_decay']**i)
        lr=self.cfg['min_lr']+(peak-self.cfg['min_lr'])*.5*(1+math.cos(math.pi*position/(self.cfg['cycle_steps']-1)))
        if i==0:lr*=min(update/self.cfg['initial_warmup_updates'],1.)
        return lr
    def step(self,update):
        if update!=self.updates+1:raise ValueError('Scheduler update discontinuity')
        lr=self.value(update)
        for group in self.optimizer.param_groups:group['lr']=lr
        self.updates=update;return lr
    def state_dict(self):return dict(updates=self.updates,config=self.cfg)
    def load_state_dict(self,state):
        if state['config']!=self.cfg:raise ValueError('Restart schedule changed during resume')
        self.updates=state['updates']

class Plateau:
    def __init__(self,cfg,cycle_steps):
        self.cfg=cfg;self.cycle_steps=cycle_steps;self.history=[];self.a_count=0;self.b_count=0;self.eval_active=False;self.stop_armed_cycle=None
    def completed_cycle(self,step,cycle_best):
        assert step%self.cycle_steps==0
        cycle=step//self.cycle_steps
        assert cycle==len(self.history)+1
        previous=min([r['cycle_best'] for r in self.history],default=None)
        improvement=None if previous is None else (previous-cycle_best)/max(abs(previous),self.cfg['epsilon'])
        eligible=step>=self.cfg['initial_training_updates'] and improvement is not None
        stop=False;started_evaluation=False
        if eligible:
            self.a_count=self.a_count+1 if improvement<self.cfg['evaluation_relative_threshold'] else 0
            self.b_count=self.b_count+1 if improvement<self.cfg['stop_relative_threshold'] else 0
            if self.a_count>=self.cfg['evaluation_patience'] and not self.eval_active:self.eval_active=True;started_evaluation=True
            if improvement>=self.cfg['stop_relative_threshold']:self.stop_armed_cycle=None
            if self.b_count>=self.cfg['stop_patience']:
                if self.stop_armed_cycle is None:self.stop_armed_cycle=cycle
                elif cycle>=self.stop_armed_cycle+1:stop=True
        record=dict(step=step,cycle=cycle,cycle_best=cycle_best,best_before=previous,relative_improvement=improvement,eligible=eligible,
            evaluation_patience_count=self.a_count,stop_patience_count=self.b_count,eval_active=self.eval_active,started_evaluation=started_evaluation,
            stop_armed_cycle=self.stop_armed_cycle,stop=stop)
        self.history.append(record);return record
    def state_dict(self):return {k:v for k,v in self.__dict__.items()}
    def load_state_dict(self,state):
        assert state['cfg']==self.cfg and state['cycle_steps']==self.cycle_steps
        self.__dict__.update(state)
