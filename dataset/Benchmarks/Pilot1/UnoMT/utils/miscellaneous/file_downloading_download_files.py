def download_files(filenames: (str or iter), target_folder: str, ftp_root:
    str=FTP_ROOT):
    """download_files(['some', 'file', 'names'], './data/, 'ftp://some-server')

    This function download one or more files from given FTP server to target
    folder. Note that the filenames wil be the same with FTP server.

    Args:
        filenames (str or iter): a string of filename or an iterable structure
            of multiple filenames for downloading.
        target_folder (str): target folder for storing downloaded data.
        ftp_root (str): address for FTP server.

    Returns:
        None
    """
    if type(filenames) is str:
        filenames = [filenames]
    try:
        os.makedirs(target_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger.error('Failed to create data folders', exc_info=True)
            raise
    for filename in filenames:
        file_path = os.path.join(target_folder, filename)
        if not os.path.exists(file_path):
            logger.debug('File does not exit. Downloading %s ...' % filename)
            url = ftp_root + filename
            try:
                url_data = urllib.request.urlopen(url)
                with open(file_path, 'wb') as f:
                    f.write(url_data.read())
            except IOError:
                logger.error('Failed to open and download url %s.' % url,
                    exc_info=True)
                raise
