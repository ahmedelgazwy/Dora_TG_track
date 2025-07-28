# In lib/train/data/loader.py

import torch
import torch.utils.data.dataloader
import collections
from lib.utils import TensorDict

string_classes = (str, bytes)

def ltr_collate(batch):
    """
    A simple collate function that stacks pre-processed and padded tensors.
    Assumes each sample in the batch has tensors of the same shape for each key.
    """
    elem = batch[0]
    if isinstance(elem, TensorDict):
        # Stack along the batch dimension
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in elem})
    
    # This is the base case
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    
    # Handle non-tensor data by returning it as a list
    if isinstance(elem, (int, float, string_classes)) or elem is None:
        return batch
        
    if isinstance(elem, collections.Mapping):
         return {key: ltr_collate([d[key] for d in batch]) for key in elem}

    # If it's a list of something else (metadata), just return it
    if isinstance(elem, (list, tuple)):
        return batch

    raise TypeError(f"Unsupported type for collate: {type(elem)}")


class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Uses a simple collate function that stacks pre-processed tensors.
    """
    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        super(LTRLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, ltr_collate, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = 0