@classmethod
def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    ret = {}
    enc_cfg = cfg['MODEL']['ENCODER']
    dec_cfg = cfg['MODEL']['DECODER']
    ret['input_shape'] = {k: v for k, v in input_shape.items() if k in
        enc_cfg['IN_FEATURES']}
    ret['conv_dim'] = enc_cfg['CONVS_DIM']
    ret['mask_dim'] = enc_cfg['MASK_DIM']
    ret['norm'] = enc_cfg['NORM']
    ret['transformer_dropout'] = dec_cfg['DROPOUT']
    ret['transformer_nheads'] = dec_cfg['NHEADS']
    ret['transformer_dim_feedforward'] = 1024
    ret['transformer_enc_layers'] = enc_cfg['TRANSFORMER_ENC_LAYERS']
    ret['transformer_in_features'] = enc_cfg[
        'DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES']
    ret['common_stride'] = enc_cfg['COMMON_STRIDE']
    return ret
