def load_pretrained_kinetics(model, pretrained_model, cfg=None,
    ignore_classifier=True, num_frames=8, num_patches=196, **kwargs):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        logging.warning(
            'Pretrained model URL is invalid, using random initialization.')
        return
    assert len(pretrained_model
        ) > 0, 'Path to pre-trained Kinetics weights not provided.'
    state_dict = load_state_dict(pretrained_model)
    classifier_name = cfg['classifier']
    if ignore_classifier:
        classifier_weight_key = classifier_name + '.weight'
        classifier_bias_key = classifier_name + '.bias'
        state_dict[classifier_weight_key] = model.state_dict()[
            classifier_weight_key]
        state_dict[classifier_bias_key] = model.state_dict()[
            classifier_bias_key]
    else:
        raise NotImplementedError(
            '[dxli] Not supporting loading Kinetics-pretrained ckpt with classifier.'
            )
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        new_pos_embed = resize_spatial_embedding(state_dict, 'pos_embed',
            num_patches)
        state_dict['pos_embed'] = new_pos_embed
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'
        ].size(1):
        state_dict['time_embed'] = resize_temporal_embedding(state_dict,
            'time_embed', num_frames)
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info('Succeeded in loading Kinetics pre-trained weights.')
    except:
        logging.error('Error in loading Kinetics pre-trained weights.')
