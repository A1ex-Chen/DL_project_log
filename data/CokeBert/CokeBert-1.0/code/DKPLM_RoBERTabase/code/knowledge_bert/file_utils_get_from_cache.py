def get_from_cache(url, cache_dir=None, force_download=False, proxies=None,
    etag_timeout=10, resume_download=False, user_agent=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if url.startswith('s3://'):
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=
                proxies, timeout=etag_timeout)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get('ETag')
        except (EnvironmentError, requests.exceptions.Timeout):
            etag = None
    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path) and etag is None:
        matching_files = [file for file in fnmatch.filter(os.listdir(
            cache_dir), filename + '.*') if not file.endswith('.json') and 
            not file.endswith('.lock')]
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])
    lock_path = cache_path + '.lock'
    with FileLock(lock_path):
        if resume_download:
            incomplete_path = cache_path + '.incomplete'

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, 'a+b') as f:
                    yield f
            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=
                cache_dir, delete=False)
            resume_size = 0
        if etag is not None and (not os.path.exists(cache_path) or
            force_download):
            with temp_file_manager() as temp_file:
                logger.info(
                    '%s not found in cache or force_download set to True, downloading to %s'
                    , url, temp_file.name)
                if url.startswith('s3://'):
                    if resume_download:
                        logger.warn(
                            'Warning: resumable downloads are not implemented for "s3://" urls'
                            )
                    s3_get(url, temp_file, proxies=proxies)
                else:
                    http_get(url, temp_file, proxies=proxies, resume_size=
                        resume_size, user_agent=user_agent)
                temp_file.flush()
                logger.info('storing %s in cache at %s', url, cache_path)
                os.rename(temp_file.name, cache_path)
                logger.info('creating metadata file for %s', cache_path)
                meta = {'url': url, 'etag': etag}
                meta_path = cache_path + '.json'
                with open(meta_path, 'w') as meta_file:
                    output_string = json.dumps(meta)
                    if sys.version_info[0] == 2 and isinstance(output_string,
                        str):
                        output_string = unicode(output_string, 'utf-8')
                    meta_file.write(output_string)
    return cache_path
