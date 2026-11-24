"""Checkpoint retention triggered by adjacent fixed-validation relative change."""
POLICY = dict(threshold=0.01, epsilon=1e-12, interval=20000,
              definition='abs(current - previous) / max(abs(previous), epsilon) < threshold',
              first_snapshot='first multiple of 20000 at or after triggering validation step',
              cadence='global step multiples of 20000; permanent once activated')


class Retention:
    def __init__(self):
        self.previous_validation = None
        self.previous_step = None
        self.trigger_step = None

    def observe(self, step, value):
        if self.previous_step is not None:
            assert step > self.previous_step
        change = None if self.previous_validation is None else abs(value-self.previous_validation)/max(abs(self.previous_validation), POLICY['epsilon'])
        triggered = self.trigger_step is None and change is not None and change < POLICY['threshold']
        if triggered:
            self.trigger_step = step
        row = dict(step=step, previous_step=self.previous_step, previous_validation=self.previous_validation,
                   current_validation=value, relative_change=change, triggered=triggered, trigger_step=self.trigger_step)
        self.previous_step, self.previous_validation = step, value
        return row

    def due(self, step):
        return self.trigger_step is not None and step >= self.trigger_step and step % POLICY['interval'] == 0

    def state_dict(self):
        return dict(policy=POLICY, previous_validation=self.previous_validation, previous_step=self.previous_step, trigger_step=self.trigger_step)

    def load_state_dict(self, state):
        assert state['policy'] == POLICY
        self.previous_validation = state['previous_validation']
        self.previous_step = state['previous_step']
        self.trigger_step = state['trigger_step']


def check():
    r = Retention()
    assert not r.due(20000)
    assert not r.observe(5000, 100.)['triggered']
    assert not r.observe(10000, 99.)['triggered']  # exactly 1% is excluded
    assert not r.observe(15000, 101.)['triggered']  # a large rise is not a small change
    assert r.observe(20000, 100.5)['triggered']
    assert r.due(20000) and not r.due(25000) and r.due(40000)
    q = Retention(); q.load_state_dict(r.state_dict())
    assert q.state_dict() == r.state_dict() and q.due(60000)
    assert not q.observe(25000, 80.)['triggered'] and q.trigger_step == 20000
    off = Retention(); off.observe(5000, 100.); off.observe(10000, 80.)
    assert off.observe(15000, 79.5)['triggered']
    assert not off.due(15000) and off.due(20000) and not off.due(35000) and off.due(40000)
    return dict(status='passed', strict_less_than_one_percent=True, absolute_change=True,
                first_snapshot_at_global_20k_boundary=True, cadence_global_20k_multiples=True, resume_state=True)
