@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.
    PathLike], *init_inputs, **kwargs):
    """
        Instantiate a :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase` (or a derived class) from
        a predefined tokenizer.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                - A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under a
                  user or organization name, like ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                  using the :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`
                  method, e.g., ``./my_model_directory/``.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  ``./my_model_directory/vocab.txt``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
                exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Attempt to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__`` method.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__`` for more details.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizerBase` so let's show our examples on a derived class: BertTokenizer
            # Download vocabulary from huggingface.co and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Download vocabulary from huggingface.co (user-uploaded) and cache.
            tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')

            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'

        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    s3_models = list(cls.max_model_input_sizes.keys())
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    vocab_files = {}
    init_configuration = {}
    if pretrained_model_name_or_path in s3_models:
        for file_id, map_list in cls.pretrained_vocab_files_map.items():
            vocab_files[file_id] = map_list[pretrained_model_name_or_path]
        if (cls.pretrained_init_configuration and 
            pretrained_model_name_or_path in cls.pretrained_init_configuration
            ):
            init_configuration = cls.pretrained_init_configuration[
                pretrained_model_name_or_path].copy()
    else:
        logger.info(
            "Model name '{}' not found in model shortcut name list ({}). Assuming '{}' is a path, a model identifier, or url to a directory containing tokenizer files."
            .format(pretrained_model_name_or_path, ', '.join(s3_models),
            pretrained_model_name_or_path))
        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
            pretrained_model_name_or_path):
            if len(cls.vocab_files_names) > 1:
                raise ValueError(
                    'Calling {}.from_pretrained() with the path to a single file or url is not supported.Use a model identifier or the path to a directory instead.'
                    .format(cls.__name__))
            logger.warning(
                'Calling {}.from_pretrained() with the path to a single file or url is deprecated'
                .format(cls.__name__))
            file_id = list(cls.vocab_files_names.keys())[0]
            vocab_files[file_id] = pretrained_model_name_or_path
        else:
            additional_files_names = {'added_tokens_file':
                ADDED_TOKENS_FILE, 'special_tokens_map_file':
                SPECIAL_TOKENS_MAP_FILE, 'tokenizer_config_file':
                TOKENIZER_CONFIG_FILE, 'tokenizer_file': FULL_TOKENIZER_FILE}
            for file_id, file_name in {**cls.vocab_files_names, **
                additional_files_names}.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    if subfolder is not None:
                        full_file_name = os.path.join(
                            pretrained_model_name_or_path, subfolder, file_name
                            )
                    else:
                        full_file_name = os.path.join(
                            pretrained_model_name_or_path, file_name)
                    if not os.path.exists(full_file_name):
                        logger.info("Didn't find file {}. We won't load it."
                            .format(full_file_name))
                        full_file_name = None
                else:
                    full_file_name = hf_bucket_url(
                        pretrained_model_name_or_path, filename=file_name,
                        subfolder=subfolder, revision=revision, mirror=None)
                vocab_files[file_id] = full_file_name
    resolved_vocab_files = {}
    for file_id, file_path in vocab_files.items():
        if file_path is None:
            resolved_vocab_files[file_id] = None
        else:
            try:
                resolved_vocab_files[file_id] = cached_path(file_path,
                    cache_dir=cache_dir, force_download=force_download,
                    proxies=proxies, resume_download=resume_download,
                    local_files_only=local_files_only)
            except requests.exceptions.HTTPError as err:
                if '404 Client Error' in str(err):
                    logger.debug(err)
                    resolved_vocab_files[file_id] = None
                else:
                    raise err
    if all(full_file_name is None for full_file_name in
        resolved_vocab_files.values()):
        msg = f"""Can't load tokenizer for '{pretrained_model_name_or_path}'. Make sure that:

- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'

- or '{pretrained_model_name_or_path}' is the correct path to a directory containing relevant tokenizer files

"""
        raise EnvironmentError(msg)
    for file_id, file_path in vocab_files.items():
        if file_path == resolved_vocab_files[file_id]:
            logger.info('loading file {}'.format(file_path))
        else:
            logger.info('loading file {} from cache at {}'.format(file_path,
                resolved_vocab_files[file_id]))
    return cls._from_pretrained(resolved_vocab_files,
        pretrained_model_name_or_path, init_configuration, *init_inputs, **
        kwargs)
