def __init__(self, config):
    super(QWenLMHeadModel, self).__init__(config)
    from .modeling_qwen import SUPPORT_BF16, logger, SUPPORT_FP16, SUPPORT_CUDA, _import_flash_attn
    autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0
    if autoset_precision:
        if SUPPORT_BF16:
            logger.warn(
                'The model is automatically converting to bf16 for faster inference. If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to "AutoModelForCausalLM.from_pretrained".'
                )
            config.bf16 = True
        elif SUPPORT_FP16:
            logger.warn(
                'The model is automatically converting to fp16 for faster inference. If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to "AutoModelForCausalLM.from_pretrained".'
                )
            config.fp16 = True
        else:
            config.fp32 = True
    if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
        logger.warn(
            'Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in "AutoModelForCausalLM.from_pretrained".'
            )
    if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
        logger.warn(
            'Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster'
            )
    if config.fp32:
        if SUPPORT_BF16:
            logger.warn(
                'Your device support faster inference by passing bf16=True in "AutoModelForCausalLM.from_pretrained".'
                )
        elif SUPPORT_FP16:
            logger.warn(
                'Your device support faster inference by passing fp16=True in "AutoModelForCausalLM.from_pretrained".'
                )
    if config.use_flash_attn == 'auto':
        if config.bf16 or config.fp16:
            logger.warn('Try importing flash-attention for faster inference...'
                )
            config.use_flash_attn = True
        else:
            config.use_flash_attn = False
    if config.use_flash_attn and config.fp32:
        logger.warn(
            'Flash attention will be disabled because it does NOT support fp32.'
            )
    if config.use_flash_attn:
        _import_flash_attn()
    self.transformer = MPLUGOwl2QWenModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    if config.bf16:
        self.transformer.bfloat16()
        self.lm_head.bfloat16()
    if config.fp16:
        self.transformer.half()
        self.lm_head.half()
    self.post_init()
