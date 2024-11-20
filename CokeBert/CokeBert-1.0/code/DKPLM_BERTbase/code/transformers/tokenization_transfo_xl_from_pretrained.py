@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *
    inputs, **kwargs):
    """
        Instantiate a pre-processed corpus.
        """
    vocab = TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path,
        *inputs, **kwargs)
    if pretrained_model_name_or_path in PRETRAINED_CORPUS_ARCHIVE_MAP:
        corpus_file = PRETRAINED_CORPUS_ARCHIVE_MAP[
            pretrained_model_name_or_path]
    else:
        corpus_file = os.path.join(pretrained_model_name_or_path, CORPUS_NAME)
    try:
        resolved_corpus_file = cached_path(corpus_file, cache_dir=cache_dir)
    except EnvironmentError:
        logger.error(
            "Corpus '{}' was not found in corpus list ({}). We assumed '{}' was a path or url but couldn't find files {} at this path or url."
            .format(pretrained_model_name_or_path, ', '.join(
            PRETRAINED_CORPUS_ARCHIVE_MAP.keys()),
            pretrained_model_name_or_path, corpus_file))
        return None
    if resolved_corpus_file == corpus_file:
        logger.info('loading corpus file {}'.format(corpus_file))
    else:
        logger.info('loading corpus file {} from cache at {}'.format(
            corpus_file, resolved_corpus_file))
    corpus = cls(*inputs, **kwargs)
    corpus_dict = torch.load(resolved_corpus_file)
    for key, value in corpus_dict.items():
        corpus.__dict__[key] = value
    corpus.vocab = vocab
    if corpus.train is not None:
        corpus.train = torch.tensor(corpus.train, dtype=torch.long)
    if corpus.valid is not None:
        corpus.valid = torch.tensor(corpus.valid, dtype=torch.long)
    if corpus.test is not None:
        corpus.test = torch.tensor(corpus.test, dtype=torch.long)
    return corpus
