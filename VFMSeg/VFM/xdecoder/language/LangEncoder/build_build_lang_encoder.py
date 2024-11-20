def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']
    if not is_lang_encoder(model_name):
        raise ValueError(f'Unkown model: {model_name}')
    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **
        kwargs)
