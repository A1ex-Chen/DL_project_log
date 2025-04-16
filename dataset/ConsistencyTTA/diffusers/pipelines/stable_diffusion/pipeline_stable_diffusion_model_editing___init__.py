def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    SchedulerMixin, safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPFeatureExtractor, requires_safety_checker: bool=
    True, with_to_k: bool=True, with_augs: list=AUGS_CONST):
    super().__init__()
    if isinstance(scheduler, PNDMScheduler):
        logger.error(
            'PNDMScheduler for this pipeline is currently not supported.')
    if safety_checker is None and requires_safety_checker:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
    self.with_to_k = with_to_k
    self.with_augs = with_augs
    ca_layers = []

    def append_ca(net_):
        if net_.__class__.__name__ == 'CrossAttention':
            ca_layers.append(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                append_ca(net__)
    for net in self.unet.named_children():
        if 'down' in net[0]:
            append_ca(net[1])
        elif 'up' in net[0]:
            append_ca(net[1])
        elif 'mid' in net[0]:
            append_ca(net[1])
    self.ca_clip_layers = [l for l in ca_layers if l.to_v.in_features == 768]
    self.projection_matrices = [l.to_v for l in self.ca_clip_layers]
    self.og_matrices = [copy.deepcopy(l.to_v) for l in self.ca_clip_layers]
    if self.with_to_k:
        self.projection_matrices = self.projection_matrices + [l.to_k for l in
            self.ca_clip_layers]
        self.og_matrices = self.og_matrices + [copy.deepcopy(l.to_k) for l in
            self.ca_clip_layers]
