@classmethod
def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=
    None, *inputs, **kwargs):
    """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
    if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
    else:
        archive_file = pretrained_model_name
    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
    except FileNotFoundError:
        logger.error(
            "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url."
            .format(pretrained_model_name, ', '.join(
            PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
        return None
    if resolved_archive_file == archive_file:
        logger.info('loading archive file {}'.format(archive_file))
    else:
        logger.info('loading archive file {} from cache at {}'.format(
            archive_file, resolved_archive_file))
    tempdir = None
    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        tempdir = tempfile.mkdtemp()
        logger.info('extracting archive file {} to temp dir {}'.format(
            resolved_archive_file, tempdir))
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path='.', members=None, *, numeric_owner=
                False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception('Attempted Path Traversal in Tar File')
                tar.extractall(path, members, numeric_owner=numeric_owner)
            safe_extract(archive, tempdir)
        serialization_dir = tempdir
    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    config = BertConfig.from_json_file(config_file)
    logger.info('Model config {}'.format(config))
    model = cls(config, *inputs, **kwargs)
    if state_dict is None:
        weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
        state_dict = torch.load(weights_path)
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-
            1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, 
            True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
    if len(missing_keys) > 0:
        logger.info('Weights of {} not initialized from pretrained model: {}'
            .format(model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info('Weights from pretrained model not used in {}: {}'.
            format(model.__class__.__name__, unexpected_keys))
    if tempdir:
        shutil.rmtree(tempdir)
    return model, missing_keys
