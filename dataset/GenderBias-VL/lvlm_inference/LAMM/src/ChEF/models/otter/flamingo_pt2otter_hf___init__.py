def __init__(self, config: OtterConfig):
    super().__init__(config)
    text_tokenizer = LlamaTokenizer.from_pretrained(config.text_config.
        _name_or_path)
    lang_encoder = LlamaForCausalLM.from_pretrained(config.text_config.
        _name_or_path)
    vision_encoder = CLIPVisionModel.from_pretrained(config.vision_config.
        _name_or_path)
    text_tokenizer.add_special_tokens({'additional_special_tokens': [
        '<|endofchunk|>', '<image>']})
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    self.text_tokenizer = text_tokenizer
    self.eoc_token_id = text_tokenizer.encode('<|endofchunk|>')[-1]
    self.media_token_id = text_tokenizer.encode('<image>')[-1]
    extend_instance(lang_encoder, OtterLMMixin)
    decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    self.lang_encoder = lang_encoder
    self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
    self.use_media_placement_augmentation = (config.
        use_media_placement_augmentation)
    self.only_attend_previous = config.only_attend_previous
    vision_encoder.output_tokens = True
    self.vision_encoder = vision_encoder
    self.vis_dim = 1024
    self.perceiver = OtterPerceiverResampler(dim=self.vis_dim)
    self.lang_encoder.init_otter(media_token_id=self.media_token_id,
        vis_hidden_size=self.vis_dim, cross_attn_every_n_layers=self.
        cross_attn_every_n_layers, use_media_placement_augmentation=self.
        use_media_placement_augmentation, only_attend_previous=self.
        only_attend_previous)
