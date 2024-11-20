@classmethod
@replace_list_option_in_docstrings()
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the :obj:`model_type` property of the config object
        that is loaded, or when it's missing, by falling back to using pattern matching on
        :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                      namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing a configuration file saved using the
                      :meth:`~transformers.PretrainedConfig.save_pretrained` method, or the
                      :meth:`~transformers.PreTrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                    - A path or url to a saved configuration JSON `file`, e.g.,
                      ``./my_model_directory/configuration.json``.
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
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs(additional keyword arguments, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        Examples::

            >>> from transformers import AutoConfig

            >>> # Download configuration from huggingface.co and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')

            >>> # Download configuration from huggingface.co (user-uploaded) and cache.
            >>> config = AutoConfig.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If configuration file is in a directory (e.g., was saved using `save_pretrained('./test/saved_model/')`).
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/')

            >>> # Load a specific configuration file.
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')

            >>> # Change some config attributes when loading a pretrained config.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            >>> config.output_attentions
            True
            >>> config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False, return_unused_kwargs=True)
            >>> config.output_attentions
            True
            >>> config.unused_kwargs
            {'foo': False}
        """
    config_dict, _ = PretrainedConfig.get_config_dict(
        pretrained_model_name_or_path, **kwargs)
    if 'model_type' in config_dict:
        config_class = CONFIG_MAPPING[config_dict['model_type']]
        return config_class.from_dict(config_dict, **kwargs)
    else:
        for pattern, config_class in CONFIG_MAPPING.items():
            if pattern in str(pretrained_model_name_or_path):
                return config_class.from_dict(config_dict, **kwargs)
    raise ValueError(
        'Unrecognized model in {}. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: {}'
        .format(pretrained_model_name_or_path, ', '.join(CONFIG_MAPPING.
        keys())))
