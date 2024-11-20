@classmethod
def get_config_dict(cls, pretrained_model_name_or_path: Union[str, os.
    PathLike], **kwargs) ->Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
    elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
        pretrained_model_name_or_path):
        config_file = pretrained_model_name_or_path
    else:
        config_file = hf_bucket_url(pretrained_model_name_or_path, filename
            =CONFIG_NAME, revision=revision, mirror=None)
    try:
        resolved_config_file = cached_path(config_file, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies, resume_download
            =resume_download, local_files_only=local_files_only)
        config_dict = cls._dict_from_json_file(resolved_config_file)
    except EnvironmentError as err:
        logger.error(err)
        msg = f"""Can't load config for '{pretrained_model_name_or_path}'. Make sure that:

- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'

- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file

"""
        raise EnvironmentError(msg)
    except json.JSONDecodeError:
        msg = (
            "Couldn't reach server at '{}' to download configuration file or configuration file is not a valid JSON file. Please check network or file content here: {}."
            .format(config_file, resolved_config_file))
        raise EnvironmentError(msg)
    if resolved_config_file == config_file:
        logger.info('loading configuration file {}'.format(config_file))
    else:
        logger.info('loading configuration file {} from cache at {}'.format
            (config_file, resolved_config_file))
    return config_dict, kwargs
