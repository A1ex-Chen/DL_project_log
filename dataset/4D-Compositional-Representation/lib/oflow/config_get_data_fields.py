def get_data_fields(mode, cfg):
    """ Returns data fields.

    Args:
        mode (str): mode (train|val|test)
        cfg (yaml config): yaml config object
    """
    fields = {}
    seq_len = cfg['data']['length_sequence']
    p_folder = cfg['data']['points_iou_seq_folder']
    pcl_folder = cfg['data']['pointcloud_seq_folder']
    mesh_folder = cfg['data']['mesh_seq_folder']
    generate_interpolate = cfg['generation']['interpolate']
    unpackbits = cfg['data']['points_unpackbits']
    transf_pt, transf_pt_val, transf_pcl, transf_pcl_val = get_transforms(cfg)
    pts_iou_field = data.PointsSubseqField
    pts_corr_field = data.PointCloudSubseqField
    if mode == 'train':
        if cfg['model']['loss_recon']:
            fields['points'] = pts_iou_field(p_folder, transform=transf_pt,
                seq_len=seq_len, fixed_time_step=0, unpackbits=unpackbits)
            fields['points_t'] = pts_iou_field(p_folder, transform=
                transf_pt, seq_len=seq_len, unpackbits=unpackbits)
        if cfg['model']['loss_corr']:
            fields['pointcloud'] = pts_corr_field(pcl_folder, transform=
                transf_pcl, seq_len=seq_len)
    elif mode == 'val':
        fields['points'] = pts_iou_field(p_folder, transform=transf_pt_val,
            all_steps=True, seq_len=seq_len, unpackbits=unpackbits)
        fields['points_mesh'] = pts_corr_field(pcl_folder, transform=
            transf_pcl_val, seq_len=seq_len)
    elif mode == 'test':
        fields['pointcloud'] = pts_corr_field(pcl_folder, transform=
            transf_pcl, seq_len=seq_len)
    return fields
