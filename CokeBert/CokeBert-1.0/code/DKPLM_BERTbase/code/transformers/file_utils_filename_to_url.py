def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError('file {} not found'.format(cache_path))
    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError('file {} not found'.format(meta_path))
    with open(meta_path, encoding='utf-8') as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']
    return url, etag
