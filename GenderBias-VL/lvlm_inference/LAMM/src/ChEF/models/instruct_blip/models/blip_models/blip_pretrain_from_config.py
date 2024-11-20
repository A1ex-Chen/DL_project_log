@classmethod
def from_config(cls, cfg=None):
    image_encoder = VisionTransformerEncoder.from_config(cfg,
        from_pretrained=True)
    text_encoder = XBertEncoder.from_config(cfg, from_pretrained=True)
    text_decoder = XBertLMHeadDecoder.from_config(cfg, from_pretrained=True)
    embed_dim = cfg.get('embed_dim', 256)
    momentum = cfg.get('momentum', 0.995)
    alpha = cfg.get('alpha', 0.4)
    max_txt_len = cfg.get('max_txt_len', 30)
    queue_size = cfg.get('queue_size', 57600)
    model = cls(image_encoder=image_encoder, text_encoder=text_encoder,
        text_decoder=text_decoder, embed_dim=embed_dim, queue_size=
        queue_size, momentum=momentum, alpha=alpha, tie_enc_dec_weights=
        True, max_txt_len=max_txt_len)
    model.reset_queue_ptr()
    return model
