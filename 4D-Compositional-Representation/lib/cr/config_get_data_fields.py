def get_data_fields(mode, cfg):
    """ Returns data fields.

    Args:
        mode (str): mode (train | val | test)å
        cfg (yaml): yaml config
    """
    fields = {}
    seq_len = cfg['data']['length_sequence']
    p_folder = cfg['data']['points_iou_seq_folder']
    transf_pt, transf_pt_val = get_transforms(cfg)
    unpackbits = cfg['data']['points_unpackbits']
    scale_type = cfg['data']['scale_type']
    pts_iou_field = data.PointsSubseqField
    if mode == 'train':
        fields['points'] = pts_iou_field(p_folder, transform=transf_pt,
            seq_len=seq_len, fixed_time_step=0, unpackbits=unpackbits,
            scale_type=scale_type)
        fields['points_t'] = pts_iou_field(p_folder, transform=transf_pt,
            seq_len=seq_len, unpackbits=unpackbits, scale_type=scale_type)
        fields['points_ex'] = pts_iou_field(p_folder, transform=transf_pt,
            seq_len=seq_len, fixed_time_step=0, unpackbits=unpackbits,
            scale_type=scale_type)
        fields['points_t_ex'] = pts_iou_field(p_folder, transform=transf_pt,
            seq_len=seq_len, unpackbits=unpackbits, scale_type=scale_type)
    elif mode == 'val':
        fields['points_iou'] = pts_iou_field(p_folder, transform=
            transf_pt_val, all_steps=True, seq_len=seq_len, unpackbits=
            unpackbits, scale_type=scale_type)
    return fields
