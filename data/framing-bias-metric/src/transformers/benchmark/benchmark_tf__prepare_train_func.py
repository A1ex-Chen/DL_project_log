def _prepare_train_func(self, model_name: str, batch_size: int,
    sequence_length: int) ->Callable[[], None]:
    config = self.config_dict[model_name]
    assert self.args.eager_mode is False, 'Training cannot be done in eager mode. Please make sure that `args.eager_mode = False`.'
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
        model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size'
        ) else config.encoder.vocab_size
    input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

    @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
    def encoder_decoder_train():
        loss = model(input_ids, decoder_input_ids=input_ids, labels=
            input_ids, training=True)[0]
        gradients = tf.gradients(loss, model.trainable_variables)
        return gradients

    @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
    def encoder_train():
        loss = model(input_ids, labels=input_ids, training=True)[0]
        gradients = tf.gradients(loss, model.trainable_variables)
        return gradients
    _train = (encoder_decoder_train if config.is_encoder_decoder else
        encoder_train)
    return _train
