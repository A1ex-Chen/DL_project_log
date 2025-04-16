@register_lang_encoder
def lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    transformer = Transformer(context_length=config_encoder[
        'CONTEXT_LENGTH'], vocab_size=tokenizer.vocab_size, width=
        config_encoder['WIDTH'], layers=config_encoder['LAYERS'], heads=
        config_encoder['HEADS'], autogressive=config_encoder.get(
        'AUTOGRESSIVE', True))
    if config_encoder.get('LOAD_PRETRAINED', False):
        transformer.load_pretrained(config_encoder['PRETRAINED'],
            config_encoder.get('PRETRAINED_LAYERS', ['*']))
    return transformer
