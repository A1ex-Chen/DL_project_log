@classmethod
@replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)
def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    """
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                      using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__()`` method.
            config (:class:`~transformers.PreTrainedConfig`, `optional`)
                The configuration object used to dertermine the tokenizer class to instantiate.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to try to load the fast version of the tokenizer.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__()`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__()`` for more details.

        Examples::

            >>> from transformers import AutoTokenizer

            >>> # Download vocabulary from huggingface.co and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            >>> tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

        """
    config = kwargs.pop('config', None)
    if not isinstance(config, PretrainedConfig):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
            **kwargs)
    use_fast = kwargs.pop('use_fast', True)
    if config.tokenizer_class is not None:
        tokenizer_class = None
        if use_fast and not config.tokenizer_class.endswith('Fast'):
            tokenizer_class_candidate = f'{config.tokenizer_class}Fast'
            tokenizer_class = tokenizer_class_from_name(
                tokenizer_class_candidate)
        if tokenizer_class is None:
            tokenizer_class_candidate = config.tokenizer_class
            tokenizer_class = tokenizer_class_from_name(
                tokenizer_class_candidate)
        if tokenizer_class is None:
            raise ValueError(
                'Tokenizer class {} does not exist or is not currently imported.'
                .format(tokenizer_class_candidate))
        return tokenizer_class.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    if isinstance(config, EncoderDecoderConfig):
        if type(config.decoder) is not type(config.encoder):
            logger.warn(
                f'The encoder model config class: {config.encoder.__class__} is different from the decoder model config class: {config.decoder.__class}. It is not recommended to use the `AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder specific tokenizer classes.'
                )
        config = config.encoder
    if type(config) in TOKENIZER_MAPPING.keys():
        tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(
            config)]
        if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
            return tokenizer_class_fast.from_pretrained(
                pretrained_model_name_or_path, *inputs, **kwargs)
        elif tokenizer_class_py is not None:
            return tokenizer_class_py.from_pretrained(
                pretrained_model_name_or_path, *inputs, **kwargs)
        else:
            raise ValueError(
                'This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed in order to use this tokenizer.'
                )
    raise ValueError(
        """Unrecognized configuration class {} to build an AutoTokenizer.
Model type should be one of {}."""
        .format(config.__class__, ', '.join(c.__name__ for c in
        TOKENIZER_MAPPING.keys())))
