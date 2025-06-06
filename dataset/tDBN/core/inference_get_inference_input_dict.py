def get_inference_input_dict(self, info, points):
    assert self.anchor_cache is not None
    assert self.target_assigner is not None
    assert self.voxel_generator is not None
    assert self.config is not None
    assert self.built is True
    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Trv2c = info['calib/Tr_velo_to_cam']
    input_cfg = self.config.eval_input_reader
    model_cfg = self.config.model.tDBN
    input_dict = {'points': points, 'rect': rect, 'Trv2c': Trv2c, 'P2': P2,
        'image_shape': np.array(info['img_shape'], dtype=np.int32),
        'image_idx': info['image_idx'], 'image_path': info['img_path']}
    out_size_factor = model_cfg.rpn.layer_strides[0
        ] // model_cfg.rpn.upsample_strides[0]
    example = prep_pointcloud(input_dict=input_dict, root_path=str(self.
        root_path), voxel_generator=self.voxel_generator, target_assigner=
        self.target_assigner, max_voxels=input_cfg.max_number_of_voxels,
        class_names=list(input_cfg.class_names), training=False,
        create_targets=False, shuffle_points=input_cfg.shuffle_points,
        generate_bev=False, without_reflectivity=model_cfg.
        without_reflectivity, num_point_features=model_cfg.
        num_point_features, anchor_area_threshold=input_cfg.
        anchor_area_threshold, anchor_cache=self.anchor_cache,
        out_size_factor=out_size_factor, out_dtype=np.float32)
    example['image_idx'] = info['image_idx']
    example['image_shape'] = input_dict['image_shape']
    example['points'] = points
    if 'anchors_mask' in example:
        example['anchors_mask'] = example['anchors_mask'].astype(np.uint8)
    example = merge_tDBN_batch([example])
    return example
