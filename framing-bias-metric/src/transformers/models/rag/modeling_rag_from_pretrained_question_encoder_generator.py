@classmethod
def from_pretrained_question_encoder_generator(cls,
    question_encoder_pretrained_model_name_or_path: str=None,
    generator_pretrained_model_name_or_path: str=None, retriever:
    RagRetriever=None, *model_args, **kwargs) ->PreTrainedModel:
    """
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            question_encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the question encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            generator_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

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
            retriever (:class:`~transformers.RagRetriever`, `optional`):
                The retriever to use.
            kwwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                ``output_attentions=True``).

                - To update the question_encoder configuration, use the prefix `question_encoder_` for each
                  configuration parameter.
                - To update the generator configuration, use the prefix `generator_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import RagModel
            >>> # initialize a RAG from two pretrained models.
            >>> model = RagModel.from_question_encoder_generator_pretrained('facebook/dpr-question_encoder-single-nq-base', 't5-small')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./rag")
            >>> # load fine-tuned model
            >>> model = RagModel.from_pretrained("./rag")

        """
    kwargs_question_encoder = {argument[len('question_question_encoder_'):]:
        value for argument, value in kwargs.items() if argument.startswith(
        'question_encoder_')}
    kwargs_generator = {argument[len('generator_'):]: value for argument,
        value in kwargs.items() if argument.startswith('generator_')}
    for key in kwargs_question_encoder.keys():
        del kwargs['question_encoder_' + key]
    for key in kwargs_generator.keys():
        del kwargs['generator_' + key]
    question_encoder = kwargs_question_encoder.pop('model', None)
    if question_encoder is None:
        assert question_encoder_pretrained_model_name_or_path is not None, 'If `model` is not defined as an argument, a `question_encoder_pretrained_model_name_or_path` has to be defined'
        from ..auto.modeling_auto import AutoModel
        if 'config' not in kwargs_question_encoder:
            from ..auto.configuration_auto import AutoConfig
            question_encoder_config = AutoConfig.from_pretrained(
                question_encoder_pretrained_model_name_or_path)
            kwargs_question_encoder['config'] = question_encoder_config
        question_encoder = AutoModel.from_pretrained(
            question_encoder_pretrained_model_name_or_path, *model_args, **
            kwargs_question_encoder)
    generator = kwargs_generator.pop('model', None)
    if generator is None:
        assert generator_pretrained_model_name_or_path is not None, 'If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has to be defined'
        from ..auto.modeling_auto import AutoModelForSeq2SeqLM
        if 'config' not in kwargs_generator:
            from ..auto.configuration_auto import AutoConfig
            generator_config = AutoConfig.from_pretrained(
                generator_pretrained_model_name_or_path)
            kwargs_generator['config'] = generator_config
        generator = AutoModelForSeq2SeqLM.from_pretrained(
            generator_pretrained_model_name_or_path, **kwargs_generator)
    config = kwargs.get('config', None)
    if config is None:
        config = RagConfig.from_question_encoder_generator_configs(
            question_encoder.config, generator.config, **kwargs)
    return cls(question_encoder=question_encoder, generator=generator,
        config=config, retriever=retriever)
