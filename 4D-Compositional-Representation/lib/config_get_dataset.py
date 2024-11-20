def get_dataset(mode, cfg, return_idx=False):
    """ Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    """
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    splits = {'train': cfg['data']['train_split'], 'val': cfg['data'][
        'val_split'], 'test': cfg['data']['test_split']}
    split = splits[mode]
    if dataset_type == 'Humans':
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field
        if return_idx:
            fields['idx'] = data.IndexField()
        dataset = data.HumansDataset(dataset_folder, fields, mode=mode,
            split=split, length_sequence=cfg['data']['length_sequence'],
            n_files_per_sequence=cfg['data']['n_files_per_sequence'],
            offset_sequence=cfg['data']['offset_sequence'], ex_folder_name=
            cfg['data']['pointcloud_seq_folder'])
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
    return dataset
