def _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups,
    zero_model_states):
    param_shapes = zero_model_states[0].param_shapes
    if debug:
        for i in range(world_size):
            for j in range(len(fp32_flat_groups[0])):
                print(
                    f'{FP32_FLAT_GROUPS}[{i}][{j}].shape={fp32_flat_groups[i][j].shape}'
                    )
    num_param_groups = len(fp32_flat_groups[0])
    merged_single_partition_of_fp32_groups = []
    for i in range(num_param_groups):
        merged_partitions = [sd[i] for sd in fp32_flat_groups]
        full_single_fp32_vector = torch.cat(merged_partitions, 0)
        merged_single_partition_of_fp32_groups.append(full_single_fp32_vector)
    avail_numel = sum([full_single_fp32_vector.numel() for
        full_single_fp32_vector in merged_single_partition_of_fp32_groups])
    if debug:
        wanted_params = sum([len(shapes) for shapes in param_shapes])
        wanted_numel = sum([sum(shape.numel() for shape in shapes.values()) for
            shapes in param_shapes])
        print(f'Have {avail_numel} numels to process.')
        print(f'Need {wanted_numel} numels in {wanted_params} params.')
    total_numel = 0
    total_params = 0
    for shapes, full_single_fp32_vector in zip(param_shapes,
        merged_single_partition_of_fp32_groups):
        offset = 0
        avail_numel = full_single_fp32_vector.numel()
        for name, shape in shapes.items():
            unpartitioned_numel = shape.numel()
            total_numel += unpartitioned_numel
            total_params += 1
            if debug:
                print(
                    f'{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} '
                    )
            state_dict[name] = full_single_fp32_vector.narrow(0, offset,
                unpartitioned_numel).view(shape)
            offset += unpartitioned_numel
        align_to = 2 * world_size

        def zero2_align(x):
            return align_to * math.ceil(x / align_to)
        if debug:
            print(f'original offset={offset}, avail_numel={avail_numel}')
        offset = zero2_align(offset)
        avail_numel = zero2_align(avail_numel)
        if debug:
            print(f'aligned  offset={offset}, avail_numel={avail_numel}')
        if offset != avail_numel:
            raise ValueError(
                f'consumed {offset} numels out of {avail_numel} - something is wrong'
                )
    print(
        f'Reconstructed fp32 state dict with {total_params} params {total_numel} elements'
        )
