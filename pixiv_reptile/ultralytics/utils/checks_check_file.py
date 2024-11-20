def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)
    file = str(file).strip()
    file = check_yolov5u_filename(file)
    if not file or '://' not in file and Path(file).exists() or file.lower(
        ).startswith('grpc://'):
        return file
    elif download and file.lower().startswith(('https://', 'http://',
        'rtsp://', 'rtmp://', 'tcp://')):
        url = file
        file = url2file(file)
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:
        files = glob.glob(str(ROOT / '**' / file), recursive=True
            ) or glob.glob(str(ROOT.parent / file))
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(
                f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []
