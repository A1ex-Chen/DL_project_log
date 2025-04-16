import copy
import os
from typing import Any, Dict, Optional, Union, Type

import torch
from torch import nn, optim


class CheckpointManager(object):
    r"""
    A :class:`CheckpointManager` periodically serializes models and optimizer as .pth files during
    training, and keeps track of best performing checkpoint based on an observed metric.

    Extended Summary
    ----------------
    It saves state dicts of models and optimizer as ``.pth`` files in a specified directory. This
    class closely follows the API of PyTorch optimizers and learning rate schedulers.

    Notes
    -----
    For :class:`~torch.nn.DataParallel` objects, ``.module.state_dict()`` is called instead of
    ``.state_dict()``.

    Parameters
    ----------
    models: Dict[str, torch.nn.Module]
        Models which need to be serialized as a checkpoint.
    optimizer: torch.optim.Optimizer
        Optimizer which needs to be serialized as a checkpoint.
    serialization_dir: str
        Path to an empty or non-existent directory to save checkpoints.
    mode: str, optional (default="max")
        One of ``min``, ``max``. In ``min`` mode, best checkpoint will be recorded when metric
        hits a lower value; in `max` mode it will be recorded when metric hits a higher value.
    filename_prefix: str, optional (default="checkpoint")
        Prefix of the to-be-saved checkpoint files.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager({"model": model}, optimizer, "/tmp/ckpt", mode="min")
    >>> num_epochs = 20
    >>> for epoch in range(num_epochs):
    ...     train(model)
    ...     val_loss = validate(model)
    ...     ckpt_manager.step(val_loss, epoch)
    """


