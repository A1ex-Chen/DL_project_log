def download_and_extract_archive(url: str, download_root: str, extract_root:
    Optional[str]=None, filename: Optional[str]=None, md5: Optional[str]=
    None, remove_finished: bool=False) ->None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)
    download_url(url, download_root, filename, md5)
    archive = os.path.join(download_root, filename)
    print('Extracting {} to {}'.format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)
