@register_body
def get_xdecoder_head(cfg, input_shape, lang_encoder, extra):
    return XDecoderHead(cfg, input_shape, lang_encoder, extra)
