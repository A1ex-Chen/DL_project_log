@register_encoder
def get_transformer_encoder_fpn(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    model = TransformerEncoderPixelDecoder(cfg, input_shape)
    forward_features = getattr(model, 'forward_features', None)
    if not callable(forward_features):
        raise ValueError(
            f'Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. Please implement forward_features for {name} to only return mask features.'
            )
    return model
