@classmethod
def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], lang_encoder:
    nn.Module, extra: dict):
    in_features_type = cfg['MODEL']['DECODER']['TRANSFORMER_IN_FEATURE']
    enc_cfg = cfg['MODEL']['ENCODER']
    dec_cfg = cfg['MODEL']['DECODER']
    if in_features_type == 'transformer_encoder':
        transformer_predictor_in_channels = enc_cfg['CONVS_DIM']
    elif in_features_type == 'pixel_embedding':
        transformer_predictor_in_channels = enc_cfg['MASK_DIM']
    elif in_features_type == 'multi_scale_pixel_decoder':
        transformer_predictor_in_channels = enc_cfg['CONVS_DIM']
    else:
        transformer_predictor_in_channels = input_shape[dec_cfg[
            'TRANSFORMER_IN_FEATURE']].channels
    return {'input_shape': {k: v for k, v in input_shape.items() if k in
        enc_cfg['IN_FEATURES']}, 'ignore_value': enc_cfg['IGNORE_VALUE'],
        'num_classes': enc_cfg.get('NUM_CLASSES', None), 'pixel_decoder':
        build_encoder(cfg, input_shape), 'loss_weight': enc_cfg[
        'LOSS_WEIGHT'], 'transformer_in_feature': dec_cfg[
        'TRANSFORMER_IN_FEATURE'], 'transformer_predictor': build_decoder(
        cfg, transformer_predictor_in_channels, lang_encoder,
        mask_classification=True, extra=extra)}
