def compute_time_ids():
    crops_coords_top_left = (args.crops_coords_top_left_h, args.
        crops_coords_top_left_w)
    original_size = target_size = args.resolution, args.resolution
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
    return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)
