def _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir):
    """
    Returns fp32 state_dict reconstructed from ds checkpoint

    Args:
        - ``ds_checkpoint_dir``: path to the deepspeed checkpoint folder (where the optimizer files are)

    """
    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")
    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files,
        ds_checkpoint_dir)
    print(
        f'Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}'
        )
    model_files = get_model_state_files(ds_checkpoint_dir)
    zero_model_states = parse_model_states(model_files)
    print(
        f'Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}'
        )
    if zero_stage <= 2:
        return _get_fp32_state_dict_from_zero2_checkpoint(world_size,
            fp32_flat_groups, zero_model_states)
    elif zero_stage == 3:
        return _get_fp32_state_dict_from_zero3_checkpoint(world_size,
            fp32_flat_groups, zero_model_states)
