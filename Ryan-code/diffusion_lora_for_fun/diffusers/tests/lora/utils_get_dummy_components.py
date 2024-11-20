def get_dummy_components(self, scheduler_cls=None, use_dora=False):
    scheduler_cls = (self.scheduler_cls if scheduler_cls is None else
        scheduler_cls)
    rank = 4
    torch.manual_seed(0)
    unet = UNet2DConditionModel(**self.unet_kwargs)
    scheduler = scheduler_cls(**self.scheduler_kwargs)
    torch.manual_seed(0)
    vae = AutoencoderKL(**self.vae_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(
        'peft-internal-testing/tiny-clip-text-2')
    tokenizer = CLIPTokenizer.from_pretrained(
        'peft-internal-testing/tiny-clip-text-2')
    if self.has_two_text_encoders:
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            'peft-internal-testing/tiny-clip-text-2')
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            'peft-internal-testing/tiny-clip-text-2')
    text_lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'out_proj'], init_lora_weights=False,
        use_dora=use_dora)
    unet_lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=[
        'to_q', 'to_k', 'to_v', 'to_out.0'], init_lora_weights=False,
        use_dora=use_dora)
    if self.has_two_text_encoders:
        pipeline_components = {'unet': unet, 'scheduler': scheduler, 'vae':
            vae, 'text_encoder': text_encoder, 'tokenizer': tokenizer,
            'text_encoder_2': text_encoder_2, 'tokenizer_2': tokenizer_2,
            'image_encoder': None, 'feature_extractor': None}
    else:
        pipeline_components = {'unet': unet, 'scheduler': scheduler, 'vae':
            vae, 'text_encoder': text_encoder, 'tokenizer': tokenizer,
            'safety_checker': None, 'feature_extractor': None,
            'image_encoder': None}
    return pipeline_components, text_lora_config, unet_lora_config
