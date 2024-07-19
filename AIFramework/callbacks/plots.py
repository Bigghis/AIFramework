from matplotlib import pyplot as plt
import fastcore.all as fc
from .callbacks import HooksCallback, Hooks, SingleBatchCB
from .plotCharts.utils import get_grid, show_image, get_hist, get_min
from .utils import to_cpu
import torch


def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, 'stats'):
        hook.stats = ([], [], [])
    acts = to_cpu(outp)
    hook.mod = mod
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40, 0, 10))


class ActivationStats(HooksCallback):
    """Callback to collect and plotting activation statistics
    """

    def __init__(self, mod_filter=fc.noop):
        """
        Args:
            mod_filter (fn, optional): . Defaults to fc.noop.
        """
        super().__init__(append_stats, mod_filter)

    def color_dim(self, figsize=(20, 5)):
        """plots neurons activation

        Args:
            figsize (tuple, optional): _description_. Defaults to (20, 5).
        """
        fig, axes = get_grid(len(self), figsize=figsize)
        index = 0
        for ax, h in zip(axes.flat, self):
            title = f'{index} {h.mod._get_name()}'
            if hasattr(h.mod, 'in_channels'):
                title = f'{title} ({h.mod.in_channels}, {h.mod.out_channels})'
            show_image(get_hist(h), ax, origin='lower',
                       noframe=False, title=title)
            index += 1

    def dead_chart(self, figsize=(11, 9)):
        """plots dead neurons, the lower the better..

        Args:
            figsize (tuple, optional): _description_. Defaults to (11, 9).
        """
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0, 1)
            title = f'{h.mod._get_name()}'
            if hasattr(h.mod, 'in_channels'):
                title = f'{title} ({h.mod.in_channels}, {h.mod.out_channels})'
            ax.set_title(title, fontsize=10)

    def plot_stats(self, figsize=(10, 4)):
        """plots means and standard deviations stats for batches

        Args:
            figsize (tuple, optional): _description_. Defaults to (10, 4).
        """
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        legends = []
        for index, h in enumerate(self):
            for i in 0, 1:
                values = h.stats[i]

                # convert to numpy array because numpy doesn't supports bf16 format, if not f32
                if values[0].dtype is torch.bfloat16:
                    values = [v.float().numpy() for v in values]
                
                axs[i].plot(values)
            title = f'{index} {h.mod._get_name()}'
            if hasattr(h.mod, 'in_channels'):
                title = f'{title} ({h.mod.in_channels}, {h.mod.out_channels})'
            legends.append(title)
        axs[0].set_title('Means', fontsize=11)
        axs[1].set_title('Stdevs', fontsize=11)
        axs[0].set_xlabel('Batch number')
        axs[0].set_ylabel('Mean')
        axs[1].set_xlabel('Batch number')
        axs[1].set_ylabel('StdDev')
        plt.legend(legends)
