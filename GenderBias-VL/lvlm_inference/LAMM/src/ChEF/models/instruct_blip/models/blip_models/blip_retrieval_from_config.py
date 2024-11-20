@classmethod
def from_config(cls, cfg=None):
    image_encoder = VisionTransformerEncoder.from_config(cfg)
    text_encoder = XBertEncoder.from_config(cfg)
    embed_dim = cfg.get('embed_dim', 256)
    momentum = cfg.get('momentum', 0.995)
    alpha = cfg.get('alpha', 0.4)
    negative_all_rank = cfg.get('negative_all_rank', False)
    queue_size = cfg.get('queue_size', 0)
    max_txt_len = cfg.get('max_txt_len', 35)
    model = cls(image_encoder=image_encoder, text_encoder=text_encoder,
        queue_size=queue_size, alpha=alpha, embed_dim=embed_dim, momentum=
        momentum, negative_all_rank=negative_all_rank, max_txt_len=max_txt_len)
    model.load_checkpoint_from_config(cfg)
    model.reset_queue_ptr()
    return model
