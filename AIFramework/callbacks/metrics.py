from copy import copy
import fastcore.all as fc
from torcheval.metrics import MulticlassAccuracy, Mean
from fastprogress import progress_bar, master_bar
from .callbacks import Callback
from .utils import to_cpu

# print metrics with Progress bar


class MetricsCB(Callback):
    '''
    This callback computes the metrics.
    '''

    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d):
        print(d)

    def before_fit(self, learn):
        learn.metrics = self

    def before_epoch(self, learn):
        [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k: f'{v.compute():.3f}' for k, v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.model.training else 'eval'
        self._log(log)

    def after_batch(self, learn):
        x, y, *_ = to_cpu(learn.batch)
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))


class ProgressCB(Callback):
    '''
    This callback prints the progress of the training.
    '''
    order = MetricsCB.order+1

    def __init__(self, plot=True):
        self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'):
            learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses:
                self.mbar.update_graph([
                    [fc.L.range(self.losses), self.losses], [fc.L.range(learn.epoch).map(
                        lambda x: (x+1)*len(learn.dataloaders.train)), self.val_losses]
                ])

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(
                    learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph([[fc.L.range(self.losses), self.losses], [fc.L.range(
                    learn.epoch+1).map(lambda x: (x+1)*len(learn.dataloaders.train)), self.val_losses]])


def get_metrics_cb():
    '''
    utility function to get the metrics callback (through MetricsCB)
    '''
    metrics = MetricsCB(accuracy=MulticlassAccuracy())
    return metrics


def get_progress_cb(plot):
    '''
    utility function to print a progress bar (through ProgressCB)
    '''
    return ProgressCB(plot)
