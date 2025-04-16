def convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_file,
    tag=None):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file that can be
    loaded with ``torch.load(file)`` + ``load_state_dict()`` and used for training without DeepSpeed.

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
        - ``output_file``: path to the pytorch fp32 state_dict output file (e.g. path/pytorch_model.bin)
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``
    """
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    print(f'Saving fp32 state dict to {output_file}')
    torch.save(state_dict, output_file)
