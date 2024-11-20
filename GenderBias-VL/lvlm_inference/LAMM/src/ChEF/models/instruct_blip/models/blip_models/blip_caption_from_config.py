@classmethod
def from_config(cls, cfg):
    image_encoder = VisionTransformerEncoder.from_config(cfg)
    text_decoder = XBertLMHeadDecoder.from_config(cfg)
    prompt = cfg.get('prompt', None)
    max_txt_len = cfg.get('max_txt_len', 40)
    model = cls(image_encoder, text_decoder, prompt=prompt, max_txt_len=
        max_txt_len)
    model.load_checkpoint_from_config(cfg)
    return model
