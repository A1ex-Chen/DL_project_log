@classmethod
def from_config(cls, cfg=None):
    image_encoder = VisionTransformerEncoder.from_config(cfg)
    text_encoder = XBertEncoder.from_config(cfg)
    use_distill = cfg.get('use_distill', True)
    momentum = cfg.get('momentum', 0.995)
    num_classes = cfg.get('num_classes', -1)
    alpha = cfg.get('alpha', 0.4)
    max_txt_len = cfg.get('max_txt_len', 40)
    assert num_classes > 1, 'Invalid number of classes provided, found {}'.format(
        num_classes)
    model = cls(image_encoder=image_encoder, text_encoder=text_encoder,
        use_distill=use_distill, alpha=alpha, num_classes=num_classes,
        momentum=momentum, max_txt_len=max_txt_len)
    pretrain_path = cfg.get('pretrained', None)
    if pretrain_path is not None:
        msg = model.load_from_pretrained(url_or_filename=pretrain_path)
    return model
