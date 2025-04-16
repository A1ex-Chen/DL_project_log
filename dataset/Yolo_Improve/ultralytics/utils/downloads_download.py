def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False,
    threads=1, retry=3, exist_ok=False):
    """
    Downloads files from specified URLs to a given directory. Supports concurrent downloads if multiple threads are
    specified.

    Args:
        url (str | list): The URL or list of URLs of the files to be downloaded.
        dir (Path, optional): The directory where the files will be saved. Defaults to the current working directory.
        unzip (bool, optional): Flag to unzip the files after downloading. Defaults to True.
        delete (bool, optional): Flag to delete the zip files after extraction. Defaults to False.
        curl (bool, optional): Flag to use curl for downloading. Defaults to False.
        threads (int, optional): Number of threads to use for concurrent downloads. Defaults to 1.
        retry (int, optional): Number of retries in case of download failure. Defaults to 3.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.

    Example:
        ```python
        download('https://ultralytics.com/assets/example.zip', dir='path/to/dir', unzip=True)
        ```
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(lambda x: safe_download(url=x[0], dir=x[1], unzip=
                unzip, delete=delete, curl=curl, retry=retry, exist_ok=
                exist_ok, progress=threads <= 1), zip(url, repeat(dir)))
            pool.close()
            pool.join()
    else:
        for u in ([url] if isinstance(url, (str, Path)) else url):
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=
                curl, retry=retry, exist_ok=exist_ok)
