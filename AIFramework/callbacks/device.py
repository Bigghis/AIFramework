import torch
import fastcore.all as fc
from typing import Mapping
from .callbacks import Callback

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)


class DeviceCB(Callback):
    def __init__(self, device):
        fc.store_attr()

    def before_fit(self, learn):
        if hasattr(learn.model, 'to'):
            learn.model.to(self.device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, device=self.device)


def get_device_cb(device):
    if device is None:
        def_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        def_device = device

    return DeviceCB(device=def_device)
