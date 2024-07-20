from .callbacks import Callback
from matplotlib import pyplot as plt
from .utilities import AccelerateCB


class BaseSchedCB(Callback):
    '''
    Base class for all schedulers.
    '''

    def __init__(self, sched):
        self.sched = sched

    def before_fit(self, learn):
        self.scheduler = self.sched(learn.opt) # first sched parameter is the optimizer!
        learn.scheduler = self.scheduler

    def _step(self, learn):
        if learn.training:  # execute scheduler only if in training mode!
            self.scheduler.step()


class BatchSchedCB(BaseSchedCB):
    """
    This callback executes the scheduler after every batch.
    """

    def after_batch(self, learn):
        self._step(learn)


class EpochSchedCB(BaseSchedCB):
    """
    This callback executes the scheduler after every epoch.
    """

    def after_epoch(self, learn):
        self._step(learn)


class HasLearnCB(Callback):
    """
    This callback stores the Learner object.
    """

    def before_fit(self, learn):
        self.learn = learn

    def after_fit(self, learn):
        self.learn = None


class RecorderCB(Callback):
    """
    This callback records scheduler's values

    example usage:
    rec = RecorderCB(lr=_lr)
    """
    order = AccelerateCB.order+1 # to use correct scheduler if acccelerateCB is setted

    def __init__(self, **d):
        self.d = d

    def before_fit(self, learn):
        self.recs = {k:
                     [] for k in self.d}
        self.pg = learn.opt.param_groups[0] # monitor torch scheduler first param_groups only

    def after_batch(self, learn):
        if not learn.training:
            return
        for k, v in self.d.items():
            self.recs[k].append(v(self))

    def plot(self):
        for k, v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()
