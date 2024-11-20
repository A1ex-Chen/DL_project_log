def build_model_from_openai_state_dict(state_dict: dict, model_cfg,
    enable_fusion: bool=False, fusion_type: str='None'):
    embed_dim = model_cfg['embed_dim']
    audio_cfg = model_cfg['audio_cfg']
    text_cfg = model_cfg['text_cfg']
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.
        startswith(f'transformer.resblocks')))
    audio_cfg = CLAPAudioCfp(**audio_cfg)
    text_cfg = CLAPTextCfg(**text_cfg)
    model = CLAP(embed_dim, audio_cfg=audio_cfg, text_cfg=text_cfg,
        quick_gelu=True, enable_fusion=enable_fusion, fusion_type=fusion_type)
    state_dict['logit_scale_a'] = state_dict['logit_scale']
    state_dict['logit_scale_t'] = state_dict['logit_scale']
    pop_keys = list(state_dict.keys())[:]
    for key in pop_keys:
        if key.startswith('visual.'):
            state_dict.pop(key, None)
    for key in ['logit_scale', 'input_resolution', 'context_length',
        'vocab_size']:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
