def load_data(args):
    data_name = args['models'][args['model']]['stage1_train_dataset']
    assert data_name in globals().keys()
    if data_name == 'LAMMDataset':
        dataset = LAMMDataset(args['data_path'], args['vision_root_path'],
            args['vision_type'])
    elif data_name == 'OctaviusDataset':
        dataset = OctaviusDataset(args['data_path_2d'], args['data_path_3d'
            ], args['vision_root_path_2d'], args['vision_root_path_3d'],
            args['loop_2d'], args['loop_3d'])
    else:
        raise ValueError(f'dataset {data_name} not found.')
    return dataset
