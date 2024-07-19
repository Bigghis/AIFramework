import numpy as np
import fastcore.all as fc
from .callbacks import Callback
from .utils import to_cpu


class BatchTransformCB(Callback):
    """ Applies a transformation to the batch during initialization.
        Used to normalize the data batch, applying mean = 0 and std = 1, for example.

        example: used to normalize data batches during initialization
    """

    def __init__(self, normalize_fn, on_train=True, on_val=True, print_means=False):
        """
        Args:
            normalize_fn (fn): function executed to normalize input data
            on_train (bool, optional): training mode. Defaults to True.
            on_val (bool, optional): validate mode. Defaults to True.
            print_means (bool, optional): if is setted print the mean and std values of the batches after every epoch. Defaults to False.
        """
        fc.store_attr()

    def after_init(self, learn):
        self.record_means = {}

    def before_epoch(self, learn):
        if self.print_means:
            self.record_means[learn.epoch] = {
                "means": [],
                "stds": []
            }

    def before_batch(self, learn):
        if (self.on_train and learn.training) or (self.on_val and not learn.training):
            learn.batch = self.normalize_fn(learn.batch)
            if self.print_means:
                self.record_means[learn.epoch]['means'].append(
                    learn.batch[0].mean())
                self.record_means[learn.epoch]['stds'].append(
                    learn.batch[0].std())

    def after_epoch(self, learn):
        if self.print_means:

            mean_mean = np.mean(
                to_cpu(self.record_means[learn.epoch]['means']))
            mean_min = np.min(to_cpu(self.record_means[learn.epoch]['means']))
            mean_max = np.max(to_cpu(self.record_means[learn.epoch]['means']))

            std_mean = np.mean(to_cpu(self.record_means[learn.epoch]['stds']))
            std_min = np.min(to_cpu(self.record_means[learn.epoch]['stds']))
            std_max = np.max(to_cpu(self.record_means[learn.epoch]['stds']))

            print(f'init stats values: Epoch {learn.epoch}: Means: (mean={mean_mean:.2f}, min={mean_min:.2f}, max={mean_max:.2f}), ' +
                  f'Stds: (mean={std_mean:.2f}, min={std_min:.2f}, max={std_max:.2f}')
