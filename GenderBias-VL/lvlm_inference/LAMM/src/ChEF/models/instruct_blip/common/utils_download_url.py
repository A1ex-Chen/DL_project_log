def download_url(url: str, root: str, filename: Optional[str]=None, md5:
    Optional[str]=None) ->None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
                                  If None, use the basename of the URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    makedir(root)
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
        return
    url = get_redirected_url(url)
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)
    try:
        print('Downloading ' + url + ' to ' + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print(
                'Failed download. Trying https -> http instead. Downloading ' +
                url + ' to ' + fpath)
            _urlretrieve(url, fpath)
        else:
            raise e
    if not check_integrity(fpath, md5):
        raise RuntimeError('File not found or corrupted.')
