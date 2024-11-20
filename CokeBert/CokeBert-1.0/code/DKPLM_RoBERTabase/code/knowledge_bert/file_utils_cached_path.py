def cached_path(url_or_filename, cache_dir=None, force_download=False,
    proxies=None, resume_download=False, user_agent=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_remote_url(url_or_filename):
        return get_from_cache(url_or_filename, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies, resume_download
            =resume_download, user_agent=user_agent)
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif urlparse(url_or_filename).scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.
            format(url_or_filename))
