import numpy as np
import math
import torch
import fastcore.all as fc
from matplotlib import pyplot as plt


@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    '''Show a PIL or PyTorch image on `ax`.'''

    if fc.hasattrs(im, ('cpu', 'permute', 'detach')):
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5:
            im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray):
        im = np.array(im)
    if im.shape[-1] == 1:
        im = im[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    # ax.set_xticks([])
    # ax.set_yticks([])
    if noframe:
        ax.axis('off')
    return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows: int = 1,  # Number of rows in returned axes grid
    ncols: int = 1,  # Number of columns in returned axes grid
    figsize: tuple = None,  # Width, height in inches of the returned figure
    # Size (in inches) of images that will be displayed in the returned figure
    imsize: int = 3,
    suptitle: str = None,  # Title to be set to returned figure
    **kwargs
):  # fig and axs
    '''A figure and set of subplots to display images of `imsize` inches'''

    if figsize is None:
        figsize = (ncols*imsize, nrows*imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows*ncols == 1:
        ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)
def get_grid(
    n: int,  # Number of axes
    nrows: int = None,  # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols: int = None,  # Number of columns, defaulting to `ceil(n/rows)`
    title: str = None,  # If passed, title set to the figure
    weight: str = 'bold',  # Title font weight
    size: int = 14,  # Title font size
    **kwargs,
):  # fig and axs
    '''Return a grid of `n` axes, `rows` by `cols`'''
    # pdb.set_trace()
    if nrows:
        ncols = ncols or int(np.floor(n/nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n/ncols))
    else:
        _nrows = int(math.sqrt(n))
        nrows = _nrows if (n % _nrows == 0) else _nrows + 1
        ncols = int(np.floor(n/_nrows))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols):
        axs.flat[i].set_axis_off()
    if title is not None:
        fig.suptitle(title, weight=weight, size=size)

    return fig, axs


def get_hist(h):
    return torch.stack(h.stats[2]).t().float().log1p()


def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)


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
