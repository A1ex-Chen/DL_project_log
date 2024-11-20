def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    """read data from KITTI-format infos, then call prep function.
    """
    v_path = pathlib.Path(root_path) / info['velodyne_path']
    v_path = v_path.parent.parent / (v_path.parent.stem + '_reduced'
        ) / v_path.name
    points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([
        -1, num_point_features])
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)
    input_dict = {'points': points, 'rect': rect, 'Trv2c': Trv2c, 'P2': P2,
        'image_shape': np.array(info['img_shape'], dtype=np.int32),
        'image_idx': image_idx, 'image_path': info['img_path']}
    if 'annos' in info:
        annos = info['annos']
        annos = kitti.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1
            ).astype(np.float32)
        difficulty = annos['difficulty']
        input_dict.update({'gt_boxes': gt_boxes, 'gt_names': gt_names,
            'difficulty': difficulty})
        if 'group_ids' in annos:
            input_dict['group_ids'] = annos['group_ids']
    example = prep_func(input_dict=input_dict)
    example['image_idx'] = image_idx
    example['image_shape'] = input_dict['image_shape']
    if 'anchors_mask' in example:
        example['anchors_mask'] = example['anchors_mask'].astype(np.uint8)
    return example
