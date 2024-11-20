@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """ Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict: key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
    if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
        config_file = cls.pretrained_config_archive_map[
            pretrained_model_name_or_path]
    elif os.path.isdir(pretrained_model_name_or_path):
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
    else:
        config_file = pretrained_model_name_or_path
    try:
        resolved_config_file = cached_path(config_file, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies)
    except EnvironmentError:
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            msg = (
                "Couldn't reach server at '{}' to download pretrained model configuration file."
                .format(config_file))
        else:
            msg = (
                "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url to a configuration file named {} or a directory containing such a file but couldn't find any such file at this path or url."
                .format(pretrained_model_name_or_path, ', '.join(cls.
                pretrained_config_archive_map.keys()), config_file,
                CONFIG_NAME))
        raise EnvironmentError(msg)
    if resolved_config_file == config_file:
        logger.info('loading configuration file {}'.format(config_file))
    else:
        logger.info('loading configuration file {} from cache at {}'.format
            (config_file, resolved_config_file))
    config = cls.from_json_file(resolved_config_file)
    if hasattr(config, 'pruned_heads'):
        config.pruned_heads = dict((int(key), value) for key, value in
            config.pruned_heads.items())
    to_remove = []
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            to_remove.append(key)
    for key in to_remove:
        kwargs.pop(key, None)
    logger.info('Model config %s', str(config))
    if return_unused_kwargs:
        return config, kwargs
    else:
        return config
