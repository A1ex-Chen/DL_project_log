def check_disk_space(url='https://ultralytics.com/assets/coco128.zip', sf=
    1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco128.zip'.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    with contextlib.suppress(Exception):
        gib = 1 << 30
        data = int(requests.head(url, timeout=20).headers['Content-Length']
            ) / gib
        _, _, free = (x / gib for x in shutil.disk_usage('/'))
        if data * sf < free:
            return True
        text = (
            f'WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, Please free {data * sf - free:.1f} GB additional disk space and try again.'
            )
        if hard:
            raise MemoryError(text)
        LOGGER.warning(text)
        return False
    return True
