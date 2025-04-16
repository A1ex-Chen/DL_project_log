def check_disk_space(url='https://ultralytics.com/assets/coco8.zip', path=
    Path.cwd(), sf=1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip'.
        path (str | Path, optional): The path or drive to check the available free space on.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    try:
        r = requests.head(url)
        assert r.status_code < 400, f'URL error for {url}: {r.status_code} {r.reason}'
    except Exception:
        return True
    gib = 1 << 30
    data = int(r.headers.get('Content-Length', 0)) / gib
    total, used, free = (x / gib for x in shutil.disk_usage(path))
    if data * sf < free:
        return True
    text = (
        f'WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, Please free {data * sf - free:.1f} GB additional disk space and try again.'
        )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False
