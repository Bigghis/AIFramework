from functools import partial
import fastcore.all as fc
from operator import attrgetter

from ..exceptions import (
    CancelFitException
)


class Callback():
    ''' Callback can be used in different points of training loop'''
    order = 0


def run_callbacks(callbacks, method_name, learn=None):
    for callback in sorted(callbacks, key=attrgetter('order')):
        method = getattr(callback, method_name, None)
        if method is not None:
            method(learn)


class Hook():
    ''' Hooks are callbacks provided by pytorch and can be used applied to a module or a tensor. '''

    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()  # remove the hook it'v very important!

    def __del__(self):
        self.remove()


class Hooks(list):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


class HooksCallback(Callback):
    def __init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None):
        """
        Args:
            hookfunc (fn): _description_
            mod_filter (_type_, optional): _description_. Defaults to fc.noop.
            on_train (bool, optional): _description_. Defaults to True.
            on_valid (bool, optional): _description_. Defaults to False.
            mods (_type_, optional): _description_. Defaults to None.
        """
        fc.store_attr()
        super().__init__()

    def before_fit(self, learn):
        if self.mods:
            mods = self.mods
        else:
            mods = fc.filter_ex(learn.model.modules(), self.mod_filter)

        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training):
            self.hookfunc(*args, **kwargs)

    def after_fit(self, learn):
        self.hooks.remove()

    def __iter__(self):
        return iter(self.hooks)

    def __len__(self):
        return len(self.hooks)


class SingleBatchCB(Callback):
    order = 1

    def after_batch(self, learn):
        raise CancelFitException()


class TrainCB(Callback):
    ''' Callback to train the model'''
    
    def __init__(self, n_inp=1): self.n_inp = n_inp

    def predict(self, learn):
        learn.preds = learn.model(*learn.batch[:self.n_inp])

    def get_loss(self, learn):
        learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad()
