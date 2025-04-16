def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)
    file = str(file).strip()
    if not file or '://' not in file and Path(file).exists():
        return file
    if download and file.lower().startswith(('https://', 'http://',
        'rtsp://', 'rtmp://')):
        url = file
        file = url2file(file)
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')
        else:
            safe_download(url=url, file=file, unzip=False)
        return file
    files = glob.glob(str(ROOT / 'cfg' / '**' / file), recursive=True)
    if not files and hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    if len(files) > 1 and hard:
        raise FileNotFoundError(
            f"Multiple files match '{file}', specify exact path: {files}")
    return files[0] if files else []
