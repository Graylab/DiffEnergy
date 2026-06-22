#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified from https://pypi.org/project/torch-fn/ by Harrison Truscott

import itertools
from typing import Callable, Optional, TypeVar
import typing
import warnings
from functools import wraps

import numpy as np
import pandas as pd
import torch

class CastWarning(RuntimeWarning): pass

def tensors(*args, **kwargs)->list[torch.Tensor]:
    return [*filter(lambda arg: isinstance(arg,torch.Tensor),args), *filter(lambda v: isinstance(v,torch.Tensor),kwargs.values())]

def is_torch(*args, **kwargs):
    return len(tensors(*args,**kwargs)) > 0


def is_cuda(*args, **kwargs):
    return any(t.is_cuda for t in tensors(*args,**kwargs))

def torch_devices(*args,**kwargs):
    return list(set(t.device for t in tensors(*args,**kwargs)))
    
def return_always(*args, **kwargs):
    return args, kwargs


def return_if(*args, **kwargs):
    if (len(args) > 0) and (len(kwargs) > 0):
        return args, kwargs
    elif (len(args) > 0) and (len(kwargs) == 0):
        return args
    elif (len(args) == 0) and (len(kwargs) > 0):
        return kwargs
    elif (len(args) == 0) and (len(kwargs) == 0):
        return None


def to_torch(args:tuple,kwargs:dict, return_fn=return_if, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _to_torch(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x_new = torch.as_tensor(x.to_numpy()).squeeze().float().to(device)
            warnings.warn(
                f"Converted from  {type(x)} to {type(x_new)} ({x_new.device})",
                category=CastWarning,
            )
            return x_new

        elif isinstance(x, (np.ndarray, list)):
            x_new = torch.as_tensor(x).float().to(device)
            warnings.warn(
                f"Converted from  {type(x)} to {type(x_new)} ({x_new.device})",
                category=CastWarning,
            )
            return x_new

        else:
            return x

    c_args = [_to_torch(arg) for arg in args]
    c_kwargs = {k: _to_torch(v) for k, v in kwargs.items()}

    # if "axis" in c_kwargs:
    #     c_kwargs["dim"] = c_kwargs.pop("axis")

    return return_fn(*c_args, **c_kwargs)


def to_numpy(args:tuple,kwargs:dict):
    def _to_numpy(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy().squeeze()
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        elif isinstance(x, tuple):
            return [_to_numpy(_x) for _x in x]
        else:
            return x

    c_args = [_to_numpy(arg) for arg in args]
    c_kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}

    # if "dim" in c_kwargs:
    #     c_kwargs["axis"] = c_kwargs.pop("dim")

    return c_args, c_kwargs

F = TypeVar("F",bound=Callable)
@typing.overload
def torch_fn(function:None=None, /, *,default_device:Optional[str|torch.device]=None,synchronize_tensors:bool=True)->Callable[[F],F]: ...
@typing.overload
def torch_fn(function:F=None, /, *,default_device:Optional[str|torch.device]=None,synchronize_tensors:bool=True)->F: ...
def torch_fn(function:Optional[Callable]=None, /, *,default_device:Optional[str|torch.device]=None,synchronize_tensors:bool=True):
    """
    Decorator to ensure torch calculation.

    Args:
        default_device: Default device on which to put new tensors. With default_device=None (default), will use cuda if available, else cpu.
        synchronize_tensors: Will examine any passed tensors for their devices. If there is consensus, will use that device 
    instead of the default_device for new tensors. If there is not consensus, raises ValueError.

    Example:
        ```import torch.nn.functional as F

        @torch_fn
        def torch_softmax(*args, **kwargs):
            return F.softmax(*args, **kwargs)

        def custom_print(x):
            print(type(x), x)

        x = [1, 2, 3]
        x_list = x
        x_tensor = torch.tensor(x).float()
        x_tensor_cuda = torch.tensor(x).float().cuda()
        x_array = np.array(x)
        x_df = pd.DataFrame({"col1": x})

        custom_print(torch_softmax(x_list, dim=-1))
        custom_print(torch_softmax(x_array, dim=-1))
        custom_print(torch_softmax(x_df, dim=-1))
        custom_print(torch_softmax(x_tensor, dim=-1))
        custom_print(torch_softmax(x_tensor_cuda, dim=-1))

        # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
        # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
        # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')```
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Buffers the input types
            is_torch_input = is_torch(*args, **kwargs)
            # is_cuda_input = is_cuda(*args, **kwargs)
            input_devices = torch_devices(*args,**kwargs)

            device = default_device
            if synchronize_tensors:
                if len(input_devices) > 1:
                    raise ValueError(f"Cannot synchronize input tensors, multiple devices found! Detected devices: {input_devices}")
                elif len(input_devices) > 0:
                    device = input_devices[0]

            # Runs the func
            c_args, c_kwargs = to_torch(args, kwargs, return_fn=return_always, device=device)

            results = func(*c_args, **c_kwargs)

            # Reverts to the original data type
            if not is_torch_input:
                return to_numpy((results,), {})[0][0]
            elif is_torch_input:
                return results
        return wrapper

    if function is not None:
        return decorator(function)
    return decorator

@typing.overload
def numpy_fn(function:None=None, /, *,require_torch_synchronize:bool=True)->Callable[[F],F]: ...
@typing.overload
def numpy_fn(function:F=None, /, *,require_torch_synchronize:bool=True)->F: ...
def numpy_fn(function,/,*,require_torch_synchronize=False):
    """
    Decorator to ensure numpy calculation.
    Args:
        require_torch_synchronize: If True, and tensors are passed to the function, will require the ouptputs to match the devices of the input devices.
        This is already done by default if all input tensors are on the same device; setting require_torch_synchronize will simply raise ValueError if the input tensors
        are on different devices.

    Example:
        ```import scipy

        @numpy_fn
        def numpy_softmax(*args, **kwargs):
            return scipy.special.softmax(*args, **kwargs)

        def custom_print(x):
            print(type(x), x)

        x = [1, 2, 3]
        x_list = x
        x_tensor = torch.tensor(x).float()
        x_tensor_cuda = torch.tensor(x).float().cuda()
        x_array = np.array(x)
        x_df = pd.DataFrame({"col1": x})

        custom_print(numpy_softmax(x_list, axis=-1))
        custom_print(numpy_softmax(x_array, axis=-1))
        custom_print(numpy_softmax(x_df, axis=-1))
        custom_print(numpy_softmax(x_tensor, axis=-1))
        custom_print(numpy_softmax(x_tensor_cuda, axis=-1))

        # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
        # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
        # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')```
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Buffers the input types
            is_torch_input = is_torch(*args, **kwargs)
            input_devices = torch_devices(*args,**kwargs)
            if len(input_devices) == 1:
                device = input_devices[0]
            elif len(input_devices) > 1 and require_torch_synchronize:
                raise ValueError(f"Cannot synchronize output tensors to inputs, inputs are on different devices! Detected devices: {input_devices}")
            else:
                device = "cpu" #don't think this should be triggerable but eh

            # Runs the func
            c_args, c_kwargs = to_numpy(args, kwargs, return_fn=return_always)
            results = func(*c_args, **c_kwargs)

            # Reverts to the original data type
            if not is_torch_input:
                return results
            elif is_torch_input:
                return to_torch((results,),{}, return_fn=return_if, device=device)[0]
        return wrapper

    if function is not None:
        return decorator(function)
    return decorator


if __name__ == "__main__":
    import scipy
    import torch.nn.functional as F

    @torch_fn
    def torch_softmax(*args, **kwargs):
        return F.softmax(*args, **kwargs)

    @numpy_fn
    def numpy_softmax(*args, **kwargs):
        return scipy.special.softmax(*args, **kwargs)

    def custom_print(x):
        print(type(x), x)

    # Test the decorator with different input types
    x = [1, 2, 3]
    x_list = x
    x_tensor = torch.tensor(x).float()
    x_tensor_cuda = torch.tensor(x).float().cuda()
    x_array = np.array(x)
    x_df = pd.DataFrame({"col1": x})

    custom_print(torch_softmax(x_list, dim=-1))
    custom_print(torch_softmax(x_array, dim=-1))
    custom_print(torch_softmax(x_df, dim=-1))
    custom_print(torch_softmax(x_tensor, dim=-1))
    custom_print(torch_softmax(x_tensor_cuda, dim=-1))
    # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
    # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
    # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')

    custom_print(numpy_softmax(x_list, axis=-1))
    custom_print(numpy_softmax(x_array, axis=-1))
    custom_print(numpy_softmax(x_df, axis=-1))
    custom_print(numpy_softmax(x_tensor, axis=-1))
    custom_print(numpy_softmax(x_tensor_cuda, axis=-1))
    # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
    # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
    # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')
