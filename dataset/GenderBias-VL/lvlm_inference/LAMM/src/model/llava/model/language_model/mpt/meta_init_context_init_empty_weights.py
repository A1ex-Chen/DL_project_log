@contextmanager
def init_empty_weights(include_buffers: bool=False):
    """Meta initialization context manager.

    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(torch.device('meta'), include_buffers=include_buffers
        ) as f:
        yield f
