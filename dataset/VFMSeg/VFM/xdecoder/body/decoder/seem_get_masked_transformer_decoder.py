@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder,
    mask_classification, extra):
    return MultiScaleMaskedTransformerDecoder(cfg, in_channels,
        lang_encoder, mask_classification, extra)
