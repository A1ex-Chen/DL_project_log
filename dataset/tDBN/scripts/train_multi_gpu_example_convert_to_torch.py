def example_convert_to_torch(example, dtype=torch.float32, device=None) ->dict:
    device = device or torch.device('cuda:0')
    example_torch = {}
    float_names = ['voxels', 'anchors', 'reg_targets', 'reg_weights',
        'bev_map', 'rect', 'Trv2c', 'P2']
    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ['coordinates', 'labels', 'num_points']:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32, device
                =device)
        elif k in ['anchors_mask']:
            example_torch[k] = torch.as_tensor(v, dtype=torch.uint8, device
                =device)
        else:
            example_torch[k] = v
    return example_torch
