@classmethod
def from_pretrained(cls, pretrained_model_name, cache_dir=None, *inputs, **
    kwargs):
    """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
    if pretrained_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP:
        vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name]
    else:
        vocab_file = pretrained_model_name
    if os.path.isdir(vocab_file):
        vocab_file = os.path.join(vocab_file, VOCAB_NAME)
    try:
        resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
    except FileNotFoundError:
        logger.error(
            "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url."
            .format(pretrained_model_name, ', '.join(
            PRETRAINED_VOCAB_ARCHIVE_MAP.keys()), vocab_file))
        return None
    if resolved_vocab_file == vocab_file:
        logger.info('loading vocabulary file {}'.format(vocab_file))
    else:
        logger.info('loading vocabulary file {} from cache at {}'.format(
            vocab_file, resolved_vocab_file))
    if (pretrained_model_name in
        PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP):
        max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[
            pretrained_model_name]
        kwargs['max_len'] = min(kwargs.get('max_len', int(1000000000000.0)),
            max_len)
    tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
    return tokenizer
