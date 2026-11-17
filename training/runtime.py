"""Untouched-base scoring must preserve trainability and random-number states."""

from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def base_model_scoring(model):
    trainability = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    def restore():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad != trainability[name]:
                parameter.requires_grad_(trainability[name])
    try:
        with model.disable_adapter():
            restore()
            with torch.no_grad():
                yield
    finally:
        restore()


@contextmanager
def preserve_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
