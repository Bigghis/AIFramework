from matplotlib import pyplot as plt
import fastcore.all as fc
from .callbacks import HooksCallback
from .plotCharts.utils import get_grid, show_image, get_hist, get_min
from .utils import to_cpu


def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, 'stats'):
        hook.stats = ([], [], [])
    acts = to_cpu(outp)
    hook.mod = mod
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40, 0, 10))


class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop):
        super().__init__(append_stats, mod_filter)

    def color_dim(self, figsize=(20, 5)):
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
        ''' plot dead neurons. The lower the better..'''
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0, 1)
            title = f'{h.mod._get_name()}'
            if hasattr(h.mod, 'in_channels'):
                title = f'{title} ({h.mod.in_channels}, {h.mod.out_channels})'
            ax.set_title(title, fontsize=10)

    def plot_stats(self, figsize=(10, 4)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        legends = []
        for index, h in enumerate(self):
            for i in 0, 1:
                axs[i].plot(h.stats[i])
            # pdb.set_trace()
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
