@classmethod
def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra
    ):
    ret = {}
    ret['lang_encoder'] = lang_encoder
    ret['in_channels'] = in_channels
    ret['mask_classification'] = mask_classification
    enc_cfg = cfg['MODEL']['ENCODER']
    dec_cfg = cfg['MODEL']['DECODER']
    ret['hidden_dim'] = dec_cfg['HIDDEN_DIM']
    ret['dim_proj'] = cfg['MODEL']['DIM_PROJ']
    ret['num_queries'] = dec_cfg['NUM_OBJECT_QUERIES']
    ret['contxt_len'] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
    ret['nheads'] = dec_cfg['NHEADS']
    ret['dim_feedforward'] = dec_cfg['DIM_FEEDFORWARD']
    assert dec_cfg['DEC_LAYERS'] >= 1
    ret['dec_layers'] = dec_cfg['DEC_LAYERS'] - 1
    ret['pre_norm'] = dec_cfg['PRE_NORM']
    ret['enforce_input_project'] = dec_cfg['ENFORCE_INPUT_PROJ']
    ret['mask_dim'] = enc_cfg['MASK_DIM']
    ret['task_switch'] = extra['task_switch']
    ret['max_spatial_len'] = dec_cfg['MAX_SPATIAL_LEN']
    ret['attn_arch'] = cfg['ATTENTION_ARCH']
    return ret
