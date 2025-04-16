def get_inputs_field(mode, cfg):
    """ Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    scale_type = None if cfg['data']['scale_type'
        ] == 'cr' and mode == 'train' else cfg['data']['scale_type']
    connected_samples = cfg['data']['input_pointcloud_corresponding']
    transform = transforms.Compose([data.SubsamplePointcloudSeq(cfg['data']
        ['input_pointcloud_n'], connected_samples=connected_samples), data.
        PointcloudNoise(cfg['data']['input_pointcloud_noise'])])
    inputs_field = data.PointCloudSubseqField(cfg['data'][
        'pointcloud_seq_folder'], transform, seq_len=cfg['data'][
        'length_sequence'], scale_type=scale_type)
    return inputs_field
