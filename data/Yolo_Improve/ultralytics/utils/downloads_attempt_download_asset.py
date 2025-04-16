def attempt_download_asset(file, repo='ultralytics/assets', release=
    'v8.2.0', **kwargs):
    """
    Attempt to download a file from GitHub release assets if it is not found locally. The function checks for the file
    locally first, then tries to download it from the specified GitHub repository release.

    Args:
        file (str | Path): The filename or file path to be downloaded.
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        release (str, optional): The specific release version to be downloaded. Defaults to 'v8.2.0'.
        **kwargs (any): Additional keyword arguments for the download process.

    Returns:
        (str): The path to the downloaded file.

    Example:
        ```python
        file_path = attempt_download_asset('yolov8n.pt', repo='ultralytics/assets', release='latest')
        ```
    """
    from ultralytics.utils import SETTINGS
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        name = Path(parse.unquote(str(file))).name
        download_url = f'https://github.com/{repo}/releases/download'
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f'Found {clean_url(url)} locally at {file}')
            else:
                safe_download(url=url, file=file, min_bytes=100000.0, **kwargs)
        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f'{download_url}/{release}/{name}', file=file,
                min_bytes=100000.0, **kwargs)
        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)
            if name in assets:
                safe_download(url=f'{download_url}/{tag}/{name}', file=file,
                    min_bytes=100000.0, **kwargs)
        return str(file)
