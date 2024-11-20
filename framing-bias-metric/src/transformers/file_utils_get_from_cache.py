def get_from_cache(url: str, cache_dir=None, force_download=False, proxies=
    None, etag_timeout=10, resume_download=False, user_agent: Union[Dict,
    str, None]=None, local_files_only=False) ->Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            headers = {'user-agent': http_user_agent(user_agent)}
            r = requests.head(url, headers=headers, allow_redirects=False,
                proxies=proxies, timeout=etag_timeout)
            r.raise_for_status()
            etag = r.headers.get('X-Linked-Etag') or r.headers.get('ETag')
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                    )
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers['Location']
        except (requests.exceptions.ConnectionError, requests.exceptions.
            Timeout):
            pass
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [file for file in fnmatch.filter(os.listdir(
                cache_dir), filename.split('.')[0] + '.*') if not file.
                endswith('.json') and not file.endswith('.lock')]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            elif local_files_only:
                raise ValueError(
                    "Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False."
                    )
            else:
                raise ValueError(
                    'Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection is on.'
                    )
    if os.path.exists(cache_path) and not force_download:
        return cache_path
    lock_path = cache_path + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if resume_download:
            incomplete_path = cache_path + '.incomplete'

            @contextmanager
            def _resumable_file_manager() ->'io.BufferedWriter':
                with open(incomplete_path, 'ab') as f:
                    yield f
            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode=
                'wb', dir=cache_dir, delete=False)
            resume_size = 0
        with temp_file_manager() as temp_file:
            logger.info(
                '%s not found in cache or force_download set to True, downloading to %s'
                , url, temp_file.name)
            http_get(url_to_download, temp_file, proxies=proxies,
                resume_size=resume_size, user_agent=user_agent)
        logger.info('storing %s in cache at %s', url, cache_path)
        os.replace(temp_file.name, cache_path)
        logger.info('creating metadata file for %s', cache_path)
        meta = {'url': url, 'etag': etag}
        meta_path = cache_path + '.json'
        with open(meta_path, 'w') as meta_file:
            json.dump(meta, meta_file)
    return cache_path
