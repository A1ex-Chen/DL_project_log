def _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups,
    zero_model_states):
    param_shapes = zero_model_states[0].param_shapes
    avail_numel = fp32_flat_groups[0].numel() * world_size
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}
    if debug:
        for i in range(world_size):
            print(f'{FP32_FLAT_GROUPS}[{i}].shape={fp32_flat_groups[i].shape}')
        wanted_params = len(param_shapes)
        wanted_numel = sum(shape.numel() for shape in param_shapes.values())
        avail_numel = fp32_flat_groups[0].numel() * world_size
        print(f'Trainable params: Have {avail_numel} numels to process.')
        print(
            f'Trainable params: Need {wanted_numel} numels in {wanted_params} params.'
            )
    offset = 0
    total_numel = 0
    total_params = 0
    for name, shape in param_shapes.items():
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1
        partitioned_numel, partitioned_padding_numel = (
            zero3_partitioned_param_info(unpartitioned_numel, world_size))
        if debug:
            print(
                f'Trainable params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}'
                )
        state_dict[name] = torch.cat(tuple(fp32_flat_groups[i].narrow(0,
            offset, partitioned_numel) for i in range(world_size)), 0).narrow(
            0, 0, unpartitioned_numel).view(shape)
        offset += partitioned_numel
    offset *= world_size
    if offset != avail_numel:
        raise ValueError(
            f'consumed {offset} numels out of {avail_numel} - something is wrong'
            )
    print(
        f'Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements'
        )
