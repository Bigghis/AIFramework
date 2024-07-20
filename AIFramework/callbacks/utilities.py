from .callbacks import Callback, TrainCB
from .device import DeviceCB
from accelerate import Accelerator
import fastcore.all as fc
import sys
import gc
import traceback
import torch


from ..exceptions import (
    CleanMemException
)


def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals():
        return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop('_i'+repr(n), None)
    user_ns.update(dict(_i='', _ii='', _iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ''


def clean_tb():
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'):
        delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'):
        delattr(sys, 'last_value')


def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CleanMemEnvironmentCB(Callback):
    """
    clear a notebook environment 
    Used when a learner is initialized in a notebook
    """
    order = 1

    def after_init(self, learn):
        try:
            clean_mem()
        except CleanMemException:
            print("clean mem failed!")


class MixedPrecisionCB(TrainCB):
    """Apply a pytorch GradScaler to scale precision from 32bit to 16bit
    and back to 32bit. This can help reduce memory footprint and speed up cuda calcs
    because Nvidia CUDA works faster with 16bit floats than 32bit floats.
    """
    order = DeviceCB.order+10

    def before_fit(self, learn):
        self.scaler = torch.cuda.amp.GradScaler()

    def before_batch(self, learn):
        self.autocast = torch.autocast("cuda", dtype=torch.float16)
        self.autocast.__enter__()

    def after_loss(self, learn):
        self.autocast.__exit__(None, None, None)

    def backward(self, learn):
        self.scaler.scale(learn.loss).backward()

    def step(self, learn):
        self.scaler.step(learn.opt)
        self.scaler.update()


class AccelerateCB(TrainCB):
    """
    Use Accelerate to handle mixed precision training

    https://huggingface.co/docs/accelerate/index

    mixed_precision="fp16" is the default mixed precision training mode. "bf16" is another option.

    ----------------------------------------------------------------------
    accelerate config example (dual GPU on local machine): 

    Which type of machine are you using? -> multi-GPU                                                                                                                                                                                                                                   
    How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                                                                                                                  
    Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: NO                                                                                                             
    Do you wish to optimize your script with torch dynamo?[yes/NO]: NO                                                                                                                                                                             
    Do you want to use DeepSpeed? [yes/NO]: NO                                                                                                                                                                                                    
    Do you want to use FullyShardedDataParallel? [yes/NO]: NO                                                                                                                                                                                      
    Do you want to use Megatron-LM ? [yes/NO]: NO                                                                                                                                                                                              
    How many GPU(s) should be used for distributed training? [1]: 2                                                                                                                                                                              
    What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
    Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
    Do you wish to use FP16 or BF16 (mixed precision)? -> fp16  

    """
    order = DeviceCB.order+11

    def __init__(self, n_inp=1, mixed_precision=None): 
        super().__init__(n_inp=n_inp)
        self.acc = Accelerator(mixed_precision=mixed_precision)

    def before_fit(self, learn):
        learn.model, learn.opt, learn.dataloaders.train, learn.dataloaders.valid, learn.scheduler = self.acc.prepare(
            learn.model, learn.opt, learn.dataloaders.train, learn.dataloaders.valid, learn.scheduler)

    def after_fit(self, learn):
        learn.model = self.acc.unwrap_model(learn.model)

    def backward(self, learn):
        self.acc.backward(learn.loss)
