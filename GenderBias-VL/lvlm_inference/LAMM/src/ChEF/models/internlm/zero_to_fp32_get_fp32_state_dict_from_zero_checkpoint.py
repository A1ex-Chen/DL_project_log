def get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=None):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated state_dict that can be loaded with
    ``load_state_dict()`` and used for training without DeepSpeed or shared with others, for example
    via a model hub.

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in 'latest' file. e.g., ``global_step14``

    Returns:
        - pytorch ``state_dict``

    Note: this approach may not work if your application doesn't have sufficient free CPU memory and
    you may need to use the offline approach using the ``zero_to_fp32.py`` script that is saved with
    the checkpoint.

    A typical usage might be ::

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        # do the training and checkpoint saving
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir) # already on cpu
        model = model.cpu() # move to cpu
        model.load_state_dict(state_dict)
        # submit to model hub or save the model to share with others

    In this example the ``model`` will no longer be usable in the deepspeed context of the same
    application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.

    If you want it all done for you, use ``load_state_dict_from_zero_checkpoint`` instead.

    """
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)
    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(
            f"Directory '{ds_checkpoint_dir}' doesn't exist")
    return _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir)
