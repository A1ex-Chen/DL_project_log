def _prepare_inference_func(self, model_name: str, batch_size: int,
    sequence_length: int) ->Callable[[], None]:
    config = self.config_dict[model_name]
    if self.args.fp16:
        raise NotImplementedError('Mixed precision is currently not supported.'
            )
    has_model_class_in_config = hasattr(config, 'architectures'
        ) and isinstance(config.architectures, list) and len(config.
        architectures) > 0
    if not self.args.only_pretrain_model and has_model_class_in_config:
        try:
            model_class = 'TF' + config.architectures[0]
            transformers_module = __import__('transformers', fromlist=[
                model_class])
            model_cls = getattr(transformers_module, model_class)
            model = model_cls(config)
        except ImportError:
            raise ImportError(
                f'{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`.'
                )
    else:
        model = TF_MODEL_MAPPING[config.__class__](config)
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size'
        ) else config.encoder.vocab_size
    input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

    @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
    def encoder_decoder_forward():
        return model(input_ids, decoder_input_ids=input_ids, training=False)

    @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
    def encoder_forward():
        return model(input_ids, training=False)
    _inference = (encoder_decoder_forward if config.is_encoder_decoder else
        encoder_forward)
    return _inference
