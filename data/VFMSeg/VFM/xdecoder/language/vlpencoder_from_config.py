@classmethod
def from_config(cls, cfg):
    tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
    tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
    lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg[
        'VERBOSE'])
    max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
    dim_lang = cfg['MODEL']['TEXT']['WIDTH']
    dim_projection = cfg['MODEL']['DIM_PROJ']
    lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
    trunc_normal_(lang_projection, std=0.02)
    return {'tokenizer': tokenizer, 'tokenizer_type': tokenizer_type,
        'lang_encoder': lang_encoder, 'lang_projection': lang_projection,
        'max_token_num': max_token_num}
