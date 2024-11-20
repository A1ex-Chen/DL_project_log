@classmethod
def from_config(cls, cfg=None):
    image_encoder = VisionTransformerEncoder.from_config(cfg)
    text_encoder = XBertEncoder.from_config(cfg)
    embed_dim = cfg.get('embed_dim', 256)
    max_txt_len = cfg.get('max_txt_len', 35)
    model = cls(image_encoder=image_encoder, text_encoder=text_encoder,
        embed_dim=embed_dim, max_txt_len=max_txt_len)
    model.load_checkpoint_from_config(cfg)
    return model
