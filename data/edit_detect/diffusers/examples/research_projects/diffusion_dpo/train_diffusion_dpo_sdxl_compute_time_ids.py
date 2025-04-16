def compute_time_ids(original_size, crops_coords_top_left):
    target_size = args.resolution, args.resolution
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    return add_time_ids
