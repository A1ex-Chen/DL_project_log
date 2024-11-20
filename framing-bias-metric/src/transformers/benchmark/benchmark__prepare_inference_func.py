def _prepare_inference_func(self, model_name: str, batch_size: int,
    sequence_length: int) ->Callable[[], None]:
    config = self.config_dict[model_name]
    if self.args.torchscript:
        config.torchscript = True
    has_model_class_in_config = hasattr(config, 'architectures'
        ) and isinstance(config.architectures, list) and len(config.
        architectures) > 0
    if not self.args.only_pretrain_model and has_model_class_in_config:
        try:
            model_class = config.architectures[0]
            transformers_module = __import__('transformers', fromlist=[
                model_class])
            model_cls = getattr(transformers_module, model_class)
            model = model_cls(config)
        except ImportError:
            raise ImportError(
                f'{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`.'
                )
    else:
        model = MODEL_MAPPING[config.__class__](config)
    model.eval()
    model.to(self.args.device)
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size'
        ) else config.encoder.vocab_size
    input_ids = torch.randint(vocab_size, (batch_size, sequence_length),
        dtype=torch.long, device=self.args.device)
    if self.args.fp16:
        logger.info('Running training in Mixed Precision...')
        assert self.args.is_gpu, 'Mixed precision is possible only for GPU.'
        model.half()
    if self.args.torchscript:
        with torch.no_grad():
            inference_model = torch.jit.trace(model, input_ids)
    else:
        inference_model = model

    def encoder_decoder_forward():
        with torch.no_grad():
            outputs = inference_model(input_ids, decoder_input_ids=input_ids)
        return outputs

    def encoder_forward():
        with torch.no_grad():
            outputs = inference_model(input_ids)
        return outputs
    _forward = (encoder_decoder_forward if config.is_encoder_decoder else
        encoder_forward)
    return _forward
