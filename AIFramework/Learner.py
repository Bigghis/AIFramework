import torch
import torch.nn.functional as F
from torch import optim
from functools import partial
import fastcore.all as fc

from .callbacks.callbacks import run_callbacks
from .exceptions import (
    CancelBatchException,
    CancelEpochException,
    CancelFitException
)


class with_cbs:
    '''
    decorator utility class used to adds callback logic to any training loop point
    in the training loop structure
    '''

    def __init__(self, loop_point):
        self.loop_point = loop_point

    def __call__(self, f):
        def fun(obj, *args, **kwargs):
            try:
                # exec before point callback
                obj.callback(f'before_{self.loop_point}')
                f(obj, *args, **kwargs)  # exec decorated function
                # exec after point callback
                obj.callback(f'after_{self.loop_point}')
            # loop point execption in globals program variables..
            except globals()[f'Cancel{self.loop_point.title()}Exception']:
                pass
            finally:
                # exec cleanup point callback
                obj.callback(f'cleanup_{self.loop_point}')
        return fun


class Learner():
    '''
    flexible learner
    '''

    def __init__(self, model, dataloaders=(0,), loss_func=F.cross_entropy, lr=0.1, callbacks=None, opt_func=optim.SGD):
        callbacks = fc.L(callbacks)
        fc.store_attr()
        self.callback = self._callback  # Assign the method here
        self.callback('after_init')

    @with_cbs('batch')
    def _one_batch(self):
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()

    @with_cbs('epoch')
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dataloaders.train if training else self.dataloaders.valid
        self._one_epoch()

    @with_cbs('fit')
    def _fit(self, train, valid):
        # training loop
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(True)
            if valid:
                torch.no_grad()(self.one_epoch)(False)

    '''
    fit function to train the model
    train True and valid True for training and validation
    train False and valid True for train only
    train False and valid True for validation only

    can bypass lr by passing it as a parameter
    and append some callbacks to the callbacks learner list
    '''

    def fit(self, n_epochs=1, train=True, valid=True, callbacks=None, lr=None):
        '''
        Fit the model for a specified number of epochs.

        Parameters:
        - n_epochs (int): Number of epochs to train (default: 1)
        - train (bool): If True, perform training (default: True)
        - valid (bool): If True, perform validation (default: True)
        - callbacks (list): Additional callbacks to use during training (default: None)
        - lr (float): Learning rate. If None, uses the lr specified during initialization (default: None)

        Behavior:
        - If train=True and valid=True: Performs both training and validation for each epoch
        - If train=True and valid=False: Performs only training for each epoch
        - If train=False and valid=True: Performs only validation for each epoch
        - If train=False and valid=False: No training or validation occurs

        '''
        callbacks = fc.L(callbacks)
        self.scheduler = None

        for callback in callbacks:
            self.callbacks.append(callback)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None:
                lr = self.lr
            if self.opt_func:
                self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        finally:
            for callback in callbacks:
                self.callbacks.remove(callback)

    def __getattr__(self, name):
        if name in ('predict', 'get_loss', 'backward', 'step', 'zero_grad'):
            return partial(self.callback, name)
        raise AttributeError(name)

    def _callback(self, method_name):
        run_callbacks(self.callbacks, method_name, self)

    @property
    def training(self):
        return self.model.training


'''
A train learner that subclass the flexible learner to implements
backward, step, zero_grad, predict, get_loss functions...
'''


class TrainLearner(Learner):
    def predict(self):
        self.preds = self.model(self.batch[0])

    def get_loss(self):
        self.loss = self.loss_func(self.preds, self.batch[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
