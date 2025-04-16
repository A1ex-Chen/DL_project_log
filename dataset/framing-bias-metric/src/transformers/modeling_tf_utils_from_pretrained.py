@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    """
        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str`, `optional`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformersTF.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch state_dict save file` (e.g, ``./pt_model/pytorch_model.bin``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            from_pt: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a PyTorch state_dict save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies: (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try doanloading the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            mirror(:obj:`str`, `optional`, defaults to :obj:`None`):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            >>> from transformers import BertConfig, TFBertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = TFBertModel.from_pretrained('bert-base-uncased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = TFBertModel.from_pretrained('./test/saved_model/')
            >>> # Update configuration during loading.
            >>> model = TFBertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a Pytorch model file instead of a TensorFlow checkpoint (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./pt_model/my_pt_model_config.json')
            >>> model = TFBertModel.from_pretrained('./pt_model/my_pytorch_model.bin', from_pt=True, config=config)

        """
    config = kwargs.pop('config', None)
    cache_dir = kwargs.pop('cache_dir', None)
    from_pt = kwargs.pop('from_pt', False)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    output_loading_info = kwargs.pop('output_loading_info', False)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    mirror = kwargs.pop('mirror', None)
    if not isinstance(config, PretrainedConfig):
        config_path = (config if config is not None else
            pretrained_model_name_or_path)
        config, model_kwargs = cls.config_class.from_pretrained(config_path,
            *model_args, cache_dir=cache_dir, return_unused_kwargs=True,
            force_download=force_download, resume_download=resume_download,
            proxies=proxies, local_files_only=local_files_only, revision=
            revision, **kwargs)
    else:
        model_kwargs = kwargs
    if pretrained_model_name_or_path is not None:
        if os.path.isdir(pretrained_model_name_or_path):
            if from_pt and os.path.isfile(os.path.join(
                pretrained_model_name_or_path, WEIGHTS_NAME)):
                archive_file = os.path.join(pretrained_model_name_or_path,
                    WEIGHTS_NAME)
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path,
                TF2_WEIGHTS_NAME)):
                archive_file = os.path.join(pretrained_model_name_or_path,
                    TF2_WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    'Error no file named {} found in directory {} or `from_pt` set to False'
                    .format([WEIGHTS_NAME, TF2_WEIGHTS_NAME],
                    pretrained_model_name_or_path))
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
            pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        elif os.path.isfile(pretrained_model_name_or_path + '.index'):
            archive_file = pretrained_model_name_or_path + '.index'
        else:
            archive_file = hf_bucket_url(pretrained_model_name_or_path,
                filename=WEIGHTS_NAME if from_pt else TF2_WEIGHTS_NAME,
                revision=revision, mirror=mirror)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=
                cache_dir, force_download=force_download, proxies=proxies,
                resume_download=resume_download, local_files_only=
                local_files_only)
        except EnvironmentError as err:
            logger.error(err)
            msg = f"""Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:

- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'

- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {TF2_WEIGHTS_NAME}, {WEIGHTS_NAME}.

"""
            raise EnvironmentError(msg)
        if resolved_archive_file == archive_file:
            logger.info('loading weights file {}'.format(archive_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(
                archive_file, resolved_archive_file))
    else:
        resolved_archive_file = None
    config.name_or_path = pretrained_model_name_or_path
    model = cls(config, *model_args, **model_kwargs)
    if from_pt:
        from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model
        return load_pytorch_checkpoint_in_tf2_model(model,
            resolved_archive_file, allow_missing_keys=True)
    model(model.dummy_inputs, training=False)
    assert os.path.isfile(resolved_archive_file
        ), 'Error retrieving file {}'.format(resolved_archive_file)
    try:
        missing_keys, unexpected_keys = load_tf_weights(model,
            resolved_archive_file)
    except OSError:
        raise OSError(
            'Unable to load weights from h5 file. If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. '
            )
    model(model.dummy_inputs, training=False)
    if cls._keys_to_ignore_on_load_missing is not None:
        for pat in cls._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is
                None]
    if cls._keys_to_ignore_on_load_unexpected is not None:
        for pat in cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat,
                k) is None]
    if len(unexpected_keys) > 0:
        logger.warning(
            f"""Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}
- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."""
            )
    else:
        logger.warning(
            f"""All model checkpoint layers were used when initializing {model.__class__.__name__}.
"""
            )
    if len(missing_keys) > 0:
        logger.warning(
            f"""Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    else:
        logger.warning(
            f"""All the layers of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.
If your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training."""
            )
    if output_loading_info:
        loading_info = {'missing_keys': missing_keys, 'unexpected_keys':
            unexpected_keys}
        return model, loading_info
    return model
