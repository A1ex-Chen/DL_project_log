@classmethod
def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    enc_cfg = cfg['MODEL']['ENCODER']
    dec_cfg = cfg['MODEL']['DECODER']
    ret = super().from_config(cfg, input_shape)
    ret['transformer_dropout'] = dec_cfg['DROPOUT']
    ret['transformer_nheads'] = dec_cfg['NHEADS']
    ret['transformer_dim_feedforward'] = dec_cfg['DIM_FEEDFORWARD']
    ret['transformer_enc_layers'] = enc_cfg['TRANSFORMER_ENC_LAYERS']
    ret['transformer_pre_norm'] = dec_cfg['PRE_NORM']
    ret['mask_on'] = cfg['MODEL']['DECODER']['MASK']
    return ret
