def build(input_reader_config, model_config, training, voxel_generator,
    target_assigner=None):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError(
            'input_reader_config not of type input_reader_pb2.InputReader.')
    generate_bev = model_config.use_bev
    without_reflectivity = model_config.without_reflectivity
    num_point_features = model_config.num_point_features
    if model_config.voxel_generator.voxel_size[0] < 0.16:
        out_size_factor = 0.4 / model_config.voxel_generator.voxel_size[0]
    else:
        out_size_factor = model_config.det_net.layer_strides[0
            ] // model_config.det_net.upsample_strides[0]
    cfg = input_reader_config
    db_sampler_cfg = input_reader_config.database_sampler
    db_sampler = None
    if len(db_sampler_cfg.sample_groups) > 0:
        db_sampler = dbsampler_builder.build(db_sampler_cfg)
    u_db_sampler_cfg = input_reader_config.unlabeled_database_sampler
    u_db_sampler = None
    if len(u_db_sampler_cfg.sample_groups) > 0:
        u_db_sampler = dbsampler_builder.build(u_db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    prep_func = partial(prep_pointcloud, root_path=cfg.kitti_root_path,
        class_names=list(cfg.class_names), voxel_generator=voxel_generator,
        target_assigner=target_assigner, training=training, max_voxels=cfg.
        max_number_of_voxels, remove_outside_points=False, remove_unknown=
        cfg.remove_unknown_examples, create_targets=training,
        shuffle_points=cfg.shuffle_points, gt_rotation_noise=list(cfg.
        groundtruth_rotation_uniform_noise), gt_loc_noise_std=list(cfg.
        groundtruth_localization_noise_std), global_rotation_noise=list(cfg
        .global_rotation_uniform_noise), global_scaling_noise=list(cfg.
        global_scaling_uniform_noise), global_random_rot_range=list(cfg.
        global_random_rotation_range_per_object), db_sampler=db_sampler,
        unlabeled_db_sampler=u_db_sampler, generate_bev=generate_bev,
        without_reflectivity=without_reflectivity, num_point_features=
        num_point_features, anchor_area_threshold=cfg.anchor_area_threshold,
        gt_points_drop=cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=cfg.remove_points_after_sample,
        remove_environment=cfg.remove_environment, use_group_id=cfg.
        use_group_id, out_size_factor=out_size_factor)
    dataset = KittiDataset(info_path=cfg.kitti_info_path, root_path=cfg.
        kitti_root_path, num_point_features=num_point_features,
        target_assigner=target_assigner, feature_map_size=feature_map_size,
        prep_func=prep_func)
    return dataset
