def cache_url(url: str, cache_dir: str) ->str:
    """
    This implementation downloads the remote resource and caches it locally.
    The resource will only be downloaded if not previously requested.
    """
    parsed_url = urlparse(url)
    dirname = os.path.join(cache_dir, os.path.dirname(parsed_url.path.
        lstrip('/')))
    makedir(dirname)
    filename = url.split('/')[-1]
    cached = os.path.join(dirname, filename)
    with file_lock(cached):
        if not os.path.isfile(cached):
            logging.info(f'Downloading {url} to {cached} ...')
            cached = download(url, dirname, filename=filename)
    logging.info(f'URL {url} cached in {cached}')
    return cached
