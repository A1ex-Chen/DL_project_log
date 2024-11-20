def merge_tDBN_batch_training(batch_list, _unused=False):
    example_merged1 = defaultdict(list)
    num_batch = len(batch_list)
    assert num_batch % 2 == 0, 'number of batch should be even'
    half = int(num_batch / 2)
    for example in batch_list[0:half]:
        for k, v in example.items():
            example_merged1[k].append(v)
    ret1 = {}
    example_merged1.pop('num_voxels')
    for key, elems in example_merged1.items():
        if key in ['voxels', 'num_points', 'num_gt', 'gt_boxes',
            'voxel_labels', 'match_indices']:
            ret1[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret1[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret1[key] = np.concatenate(coors, axis=0)
        else:
            ret1[key] = np.stack(elems, axis=0)
    example_merged2 = defaultdict(list)
    for example in batch_list[half:]:
        for k, v in example.items():
            example_merged2[k].append(v)
    ret2 = {}
    example_merged2.pop('num_voxels')
    for key, elems in example_merged2.items():
        if key in ['voxels', 'num_points', 'num_gt', 'gt_boxes',
            'voxel_labels', 'match_indices']:
            ret2[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret2[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret2[key] = np.concatenate(coors, axis=0)
        else:
            ret2[key] = np.stack(elems, axis=0)
    ret = ret1, ret2
    return ret
