def parse_optim_states(files, ds_checkpoint_dir):
    total_files = len(files)
    state_dicts = []
    for f in files:
        state_dict = torch.load(f, map_location=device)
        state_dict['optimizer_state_dict'].pop('optimizer_state_dict', None)
        state_dicts.append(state_dict)
    if not ZERO_STAGE in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f'{files[0]} is not a zero checkpoint')
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]
    if type(world_size) is list:
        world_size = max(world_size)
    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
            )
    if zero_stage <= 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f'unknown zero stage {zero_stage}')
    if zero_stage <= 2:
        fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][
            fp32_groups_key] for i in range(len(state_dicts))]
    elif zero_stage == 3:
        fp32_flat_groups = [torch.cat(state_dicts[i][OPTIMIZER_STATE_DICT][
            fp32_groups_key], 0) for i in range(len(state_dicts))]
    return zero_stage, world_size, fp32_flat_groups
