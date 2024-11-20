@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    """Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch state_dict save file` (e.g. `./pt_model/pytorch_model.bin`). In this case, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the PyTorch checkpoint in a TensorFlow model using the provided conversion scripts and loading the TensorFlow model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            from_pt: (`optional`) boolean, default False:
                Load the model weights from a PyTorch state_dict save file (see docstring of pretrained_model_name_or_path argument).

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_pt=True, config=config)

        """
    config = kwargs.pop('config', None)
    cache_dir = kwargs.pop('cache_dir', None)
    from_pt = kwargs.pop('from_pt', False)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    if config is None:
        config, model_kwargs = cls.config_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, cache_dir=cache_dir,
            return_unused_kwargs=True, force_download=force_download, **kwargs)
    else:
        model_kwargs = kwargs
    if pretrained_model_name_or_path is not None:
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[
                pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path,
                TF2_WEIGHTS_NAME)):
                archive_file = os.path.join(pretrained_model_name_or_path,
                    TF2_WEIGHTS_NAME)
            elif from_pt and os.path.isfile(os.path.join(
                pretrained_model_name_or_path, WEIGHTS_NAME)):
                archive_file = os.path.join(pretrained_model_name_or_path,
                    WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    'Error no file named {} found in directory {} or `from_pt` set to False'
                    .format([WEIGHTS_NAME, TF2_WEIGHTS_NAME],
                    pretrained_model_name_or_path))
        elif os.path.isfile(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        else:
            raise EnvironmentError('Error file {} not found'.format(
                pretrained_model_name_or_path))
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=
                cache_dir, force_download=force_download, proxies=proxies)
        except EnvironmentError as e:
            if (pretrained_model_name_or_path in cls.
                pretrained_model_archive_map):
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights."
                    .format(archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url."
                    .format(pretrained_model_name_or_path, ', '.join(cls.
                    pretrained_model_archive_map.keys()), archive_file))
            raise e
        if resolved_archive_file == archive_file:
            logger.info('loading weights file {}'.format(archive_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(
                archive_file, resolved_archive_file))
    else:
        resolved_archive_file = None
    model = cls(config, *model_args, **model_kwargs)
    if from_pt:
        return load_pytorch_checkpoint_in_tf2_model(model,
            resolved_archive_file)
    ret = model(model.dummy_inputs, training=False)
    assert os.path.isfile(resolved_archive_file
        ), 'Error retrieving file {}'.format(resolved_archive_file)
    model.load_weights(resolved_archive_file, by_name=True)
    ret = model(model.dummy_inputs, training=False)
    return model
