def get_data_fields(cfg, mode='train'):
    """ Returns the data fields.

    Args:
        cfg (dict): imported yaml config
        mode (str): the mode which is used

    Return:
        field (dict): datafield
    """
    use_DPT = cfg['depth']['type'] == 'DPT'
    resize_img_transform = ResizeImage_mvs()
    fields = {}
    load_ref_img = cfg['training']['pc_weight'] != 0.0 or cfg['training'][
        'rgb_s_weight'] != 0.0
    dataset_name = cfg['dataloading']['dataset_name']
    if dataset_name == 'any':
        img_field = DataField(model_path=cfg['dataloading']['path'],
            transform=resize_img_transform, with_camera=True, with_depth=
            cfg['dataloading']['with_depth'], scene_name=cfg['dataloading']
            ['scene'], use_DPT=use_DPT, mode=mode, spherify=cfg[
            'dataloading']['spherify'], load_ref_img=load_ref_img,
            customized_poses=cfg['dataloading']['customized_poses'],
            customized_focal=cfg['dataloading']['customized_focal'],
            resize_factor=cfg['dataloading']['resize_factor'], depth_net=
            cfg['dataloading']['depth_net'], crop_size=cfg['dataloading'][
            'crop_size'], random_ref=cfg['dataloading']['random_ref'],
            norm_depth=cfg['dataloading']['norm_depth'], load_colmap_poses=
            cfg['dataloading']['load_colmap_poses'], sample_rate=cfg[
            'dataloading']['sample_rate'])
    else:
        print(dataset_name, 'does not exist')
    fields['img'] = img_field
    return fields
