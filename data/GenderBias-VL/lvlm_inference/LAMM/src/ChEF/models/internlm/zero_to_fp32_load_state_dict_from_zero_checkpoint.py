def load_state_dict_from_zero_checkpoint(model, checkpoint_dir, tag=None):
    """
    1. Put the provided model to cpu
    2. Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict``
    3. Load it into the provided model

    Args:
        - ``model``: the model object to update
        - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``

    Returns:
        - ``model`: modified model

    Make sure you have plenty of CPU memory available before you call this function. If you don't
    have enough use the ``zero_to_fp32.py`` utility to do the conversion. You will find it
    conveniently placed for you in the checkpoint folder.

    A typical usage might be ::

        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
        # submit to model hub or save the model to share with others

    Note, that once this was run, the ``model`` will no longer be usable in the deepspeed context
    of the same application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.

    """
    logger.info(f'Extracting fp32 weights')
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    logger.info(f'Overwriting model with fp32 weights')
    model = model.cpu()
    model.load_state_dict(state_dict, strict=False)
    return model
