def pipeline(task: str, model: Optional=None, config: Optional[Union[str,
    PretrainedConfig]]=None, tokenizer: Optional[Union[str,
    PreTrainedTokenizer]]=None, framework: Optional[str]=None, revision:
    Optional[str]=None, use_fast: bool=True, **kwargs) ->Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`.
            - :obj:`"sentiment-analysis"`: will return a :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"ner"`: will return a :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`.
            - :obj:`"fill-mask"`: will return a :class:`~transformers.FillMaskPipeline`.
            - :obj:`"summarization"`: will return a :class:`~transformers.SummarizationPipeline`.
            - :obj:`"translation_xx_to_yy"`: will return a :class:`~transformers.TranslationPipeline`.
            - :obj:`"text2text-generation"`: will return a :class:`~transformers.Text2TextGenerationPipeline`.
            - :obj:`"text-generation"`: will return a :class:`~transformers.TextGenerationPipeline`.
            - :obj:`"zero-shot-classification:`: will return a :class:`~transformers.ZeroShotClassificationPipeline`.
            - :obj:`"conversation"`: will return a :class:`~transformers.ConversationalPipeline`.
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from :class:`~transformers.PreTrainedModel` (for PyTorch)
            or :class:`~transformers.TFPreTrainedModel` (for TensorFlow).

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default configuration file for the requested model will be used. That means that if
            :obj:`model` is given, its default configuration will be used. However, if :obj:`model` is not supplied,
            this :obj:`task`'s default model's config is used instead.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default tokenizer for the given :obj:`model` will be loaded (if it is a string). If
            :obj:`model` is not specified or not a string, then the default tokenizer for :obj:`config` is loaded (if
            it is a string). However, if :obj:`config` is also not given or not a string, then the default tokenizer
            for the given :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.
        use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use a Fast tokenizer if possible (a :class:`~transformers.PreTrainedTokenizerFast`).
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        >>> # Sentiment analysis pipeline
        >>> pipeline('sentiment-analysis')

        >>> # Question answering pipeline, specifying the checkpoint identifier
        >>> pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
        >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> pipeline('ner', model=model, tokenizer=tokenizer)
    """
    targeted_task, task_options = check_task(task)
    if model is None:
        model = get_default_model(targeted_task, framework, task_options)
    framework = framework or get_framework(model)
    task_class, model_class = targeted_task['impl'], targeted_task[framework]
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            raise Exception(
                'Impossible to guess which tokenizer to use. Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer.'
                )
    modelcard = None
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            use_fast = tokenizer[1].pop('use_fast', use_fast)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0],
                use_fast=use_fast, revision=revision, **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=
                revision, use_fast=use_fast)
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, revision=revision)
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard, revision=revision)
    if isinstance(model, str):
        model_kwargs = {}
        if framework == 'pt' and model.endswith('.h5'):
            model_kwargs['from_tf'] = True
            logger.warning(
                'Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. Trying to load the model with PyTorch.'
                )
        elif framework == 'tf' and model.endswith('.bin'):
            model_kwargs['from_pt'] = True
            logger.warning(
                'Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. Trying to load the model with Tensorflow.'
                )
        model = model_class.from_pretrained(model, config=config, revision=
            revision, **model_kwargs)
        if task == 'translation' and model.config.task_specific_params:
            for key in model.config.task_specific_params:
                if key.startswith('translation'):
                    task = key
                    warnings.warn(
                        '"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{}"'
                        .format(task), UserWarning)
                    break
    return task_class(model=model, tokenizer=tokenizer, modelcard=modelcard,
        framework=framework, task=task, **kwargs)
