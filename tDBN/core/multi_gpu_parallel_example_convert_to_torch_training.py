def example_convert_to_torch_training(example, dtype=torch.float32, device=None
    ) ->dict:
    example_torch1 = {}
    float_names = ['voxels', 'anchors', 'reg_targets', 'reg_weights',
        'bev_map', 'rect', 'Trv2c', 'P2']
    for k, v in example[0].items():
        if k in float_names:
            example_torch1[k] = torch.as_tensor(v, dtype=dtype, device=0)
        elif k in ['coordinates', 'labels', 'num_points']:
            example_torch1[k] = torch.as_tensor(v, dtype=torch.int32, device=0)
        elif k in ['anchors_mask']:
            example_torch1[k] = torch.as_tensor(v, dtype=torch.uint8, device=0)
        else:
            example_torch1[k] = v
    example_torch2 = {}
    for k, v in example[1].items():
        if k in float_names:
            example_torch2[k] = torch.as_tensor(v, dtype=dtype, device=1)
        elif k in ['coordinates', 'labels', 'num_points']:
            example_torch2[k] = torch.as_tensor(v, dtype=torch.int32, device=1)
        elif k in ['anchors_mask']:
            example_torch2[k] = torch.as_tensor(v, dtype=torch.uint8, device=1)
        else:
            example_torch2[k] = v
    example_torch = tuple([example_torch1, example_torch2])
    return example_torch
