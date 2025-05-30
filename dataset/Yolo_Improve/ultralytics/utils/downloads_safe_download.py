def safe_download(url, file=None, dir=None, unzip=True, delete=False, curl=
    False, retry=3, min_bytes=1.0, exist_ok=False, progress=True):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.

    Example:
        ```python
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```
    """
    gdrive = url.startswith('https://drive.google.com/')
    if gdrive:
        url, file = get_google_drive_file_info(url)
    f = Path(dir or '.') / (file or url2file(url))
    if '://' not in str(url) and Path(url).is_file():
        f = Path(url)
    elif not f.is_file():
        uri = (url if gdrive else clean_url(url)).replace(
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/',
            'https://ultralytics.com/assets/')
        desc = f"Downloading {uri} to '{f}'"
        LOGGER.info(f'{desc}...')
        f.parent.mkdir(parents=True, exist_ok=True)
        check_disk_space(url, path=f.parent)
        for i in range(retry + 1):
            try:
                if curl or i > 0:
                    s = 'sS' * (not progress)
                    r = subprocess.run(['curl', '-#', f'-{s}L', url, '-o',
                        f, '--retry', '3', '-C', '-']).returncode
                    assert r == 0, f'Curl return value {r}'
                else:
                    method = 'torch'
                    if method == 'torch':
                        torch.hub.download_url_to_file(url, f, progress=
                            progress)
                    else:
                        with request.urlopen(url) as response, TQDM(total=
                            int(response.getheader('Content-Length', 0)),
                            desc=desc, disable=not progress, unit='B',
                            unit_scale=True, unit_divisor=1024) as pbar:
                            with open(f, 'wb') as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))
                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break
                    f.unlink()
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(
                        f'❌  Download failure for {uri}. Environment is not online.'
                        )) from e
                elif i >= retry:
                    raise ConnectionError(emojis(
                        f'❌  Download failure for {uri}. Retry limit reached.')
                        ) from e
                LOGGER.warning(
                    f'⚠️ Download failure, retrying {i + 1}/{retry} {uri}...')
    if unzip and f.exists() and f.suffix in {'', '.zip', '.tar', '.gz'}:
        from zipfile import is_zipfile
        unzip_dir = (dir or f.parent).resolve()
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=
                exist_ok, progress=progress)
        elif f.suffix in {'.tar', '.gz'}:
            LOGGER.info(f'Unzipping {f} to {unzip_dir}...')
            subprocess.run(['tar', 'xf' if f.suffix == '.tar' else 'xfz', f,
                '--directory', unzip_dir], check=True)
        if delete:
            f.unlink()
        return unzip_dir
