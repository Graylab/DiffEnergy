import torch

def squeeze_batch(batch):
    """
    Squeeze the batch to remove unnecessary dimensions.
    """
    if isinstance(batch, torch.Tensor):
        return batch.squeeze()
    elif isinstance(batch, list):
        return [squeeze_batch(b) for b in batch]
    elif isinstance(batch, dict):
        return {k: squeeze_batch(v) for k, v in batch.items()}
    else:
        raise TypeError("Unsupported batch type: {}".format(type(batch)))

