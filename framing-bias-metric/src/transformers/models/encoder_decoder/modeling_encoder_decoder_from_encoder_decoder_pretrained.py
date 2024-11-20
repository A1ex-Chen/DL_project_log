@classmethod
def from_encoder_decoder_pretrained(cls,
    encoder_pretrained_model_name_or_path: str=None,
    decoder_pretrained_model_name_or_path: str=None, *model_args, **kwargs
    ) ->PreTrainedModel:
    """
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.

            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`).

                - To update the encoder configuration, use the prefix `encoder_` for each configuration parameter.
                - To update the decoder configuration, use the prefix `decoder_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import EncoderDecoderModel
            >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert2bert")
            >>> # load fine-tuned model
            >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")

        """
    kwargs_encoder = {argument[len('encoder_'):]: value for argument, value in
        kwargs.items() if argument.startswith('encoder_')}
    kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in
        kwargs.items() if argument.startswith('decoder_')}
    for key in kwargs_encoder.keys():
        del kwargs['encoder_' + key]
    for key in kwargs_decoder.keys():
        del kwargs['decoder_' + key]
    encoder = kwargs_encoder.pop('model', None)
    if encoder is None:
        assert encoder_pretrained_model_name_or_path is not None, 'If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined'
        from ..auto.modeling_auto import AutoModel
        if 'config' not in kwargs_encoder:
            from ..auto.configuration_auto import AutoConfig
            encoder_config = AutoConfig.from_pretrained(
                encoder_pretrained_model_name_or_path)
            if (encoder_config.is_decoder is True or encoder_config.
                add_cross_attention is True):
                logger.info(
                    f'Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled.'
                    )
                encoder_config.is_decoder = False
                encoder_config.add_cross_attention = False
            kwargs_encoder['config'] = encoder_config
        encoder = AutoModel.from_pretrained(
            encoder_pretrained_model_name_or_path, *model_args, **
            kwargs_encoder)
    decoder = kwargs_decoder.pop('model', None)
    if decoder is None:
        assert decoder_pretrained_model_name_or_path is not None, 'If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined'
        from ..auto.modeling_auto import AutoModelForCausalLM
        if 'config' not in kwargs_decoder:
            from ..auto.configuration_auto import AutoConfig
            decoder_config = AutoConfig.from_pretrained(
                decoder_pretrained_model_name_or_path)
            if (decoder_config.is_decoder is False or decoder_config.
                add_cross_attention is False):
                logger.info(
                    f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True
            kwargs_decoder['config'] = decoder_config
        if kwargs_decoder['config'].is_decoder is False or kwargs_decoder[
            'config'].add_cross_attention is False:
            logger.warning(
                f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`'
                )
        decoder = AutoModelForCausalLM.from_pretrained(
            decoder_pretrained_model_name_or_path, **kwargs_decoder)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.
        config, decoder.config, **kwargs)
    return cls(encoder=encoder, decoder=decoder, config=config)
