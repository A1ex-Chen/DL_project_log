def fetch_or_cache_model(url: str, dst: (PathLike | None)=None, hash_prefix:
    (str | None)=None, progress: bool=True) ->str:
    """Fetches a model from a URL or uses a cached version if it exists.

    Args:
        url (str): URL of the object to download.
        dst (PathLike | None, optional): Full path where object will be saved. Defaults to None.
        hash_prefix (str | None, optional): Hash prefix to validate downloaded file. Defaults to None.
        progress (bool, optional): Whether to show download progress. Defaults to True.

    Returns:
        str: The path to the downloaded or cached file.
    """
    CACHE_DIR = Path('~/.cache/sportslabkit').expanduser()
    hashed_url = hashlib.sha256(url.encode()).hexdigest()
    if Path(url).exists():
        return url
    if dst is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dst = CACHE_DIR / f'{sanitize_url_name(url)}_{hashed_url}'
    if os.path.exists(dst):
        return str(dst)
    if url.startswith('https://drive.google.com'):
        if os.path.exists(dst):
            return str(dst)
        gdown.download(str(url), str(dst), quiet=False, fuzzy=True)
        assert os.path.exists(dst)
        return str(dst)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.partial') as f:
        tmp_dst = f.name
    try:
        req = Request(url, headers={'User-Agent': 'sportslabkit'})
        u = urlopen(req)
        meta = u.info()
        file_size = int(meta.get_all('Content-Length')[0]) if meta.get_all(
            'Content-Length') else None
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress, desc=
            f'Downloading file to {dst}', unit='B', unit_scale=True,
            unit_divisor=1024) as pbar:
            with open(tmp_dst, 'wb') as f:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if hash_prefix is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    f'Invalid hash value (expected "{hash_prefix}", got "{digest}")'
                    )
        shutil.move(tmp_dst, dst)
    finally:
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)
    return str(dst)
