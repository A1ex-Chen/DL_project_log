def cached_path(url_or_filename, cache_dir=None, force_download=False,
    proxies=None, resume_download=False, user_agent: Union[Dict, str, None]
    =None, extract_compressed_file=False, force_extract=False,
    local_files_only=False) ->Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_remote_url(url_or_filename):
        output_path = get_from_cache(url_or_filename, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies, resume_download
            =resume_download, user_agent=user_agent, local_files_only=
            local_files_only)
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.
            format(url_or_filename))
    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace('.', '-') + '-extracted'
        output_path_extracted = os.path.join(output_dir,
            output_extract_dir_name)
        if os.path.isdir(output_path_extracted) and os.listdir(
            output_path_extracted) and not force_extract:
            return output_path_extracted
        lock_path = output_path + '.lock'
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, 'r') as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(
                    'Archive format of {} could not be identified'.format(
                    output_path))
        return output_path_extracted
    return output_path
