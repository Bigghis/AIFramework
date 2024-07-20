import torch
import torch.nn.functional as F
from torch import optim
import fastcore.all as fc

from .callbacks.callbacks import run_callbacks
from .exceptions import (
    CancelBatchException,
    CancelEpochException,
    CancelFitException
)


class Learner():
    def __init__(self, model, dataloaders, lr, callbacks, force_train=False, force_eval=False, loss_func=F.cross_entropy, opt_func=optim.SGD):
        fc.store_attr()
        self.callback('after_init')

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

    def one_batch(self):
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.model.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()

    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dataloaders.train if train else self.dataloaders.valid
        try:
            self.callback('before_epoch')
            for self.iter, self.batch in enumerate(self.dl):
                try:
                    self.callback('before_batch')
                    self.one_batch()
                    self.callback('after_batch')
                except CancelBatchException:
                    pass
            self.callback('after_epoch')
        except CancelEpochException:
            pass

    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self.scheduler = None
        try:
            self.callback('before_fit')
            for self.epoch in self.epochs:
                if self.force_train: # force training only
                    self.one_epoch(True)
                elif self.force_eval: # force eval only
                    self.one_epoch(False)
                else:  # default case --> training + eval for every epoch
                    self.one_epoch(True)
                    self.one_epoch(False)
            self.callback('after_fit')
        except CancelFitException:
            pass

    def callback(self, method_nm):
        run_callbacks(self.callbacks, method_nm, self)

    @property
    def training(self):
        return self.model.training
