def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split('/')[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f'{download_target} exists and is not a regular file')
    if os.path.isfile(download_target):
        if insecure_hashlib.sha256(open(download_target, 'rb').read()
            ).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f'{download_target} exists, but the SHA256 checksum does not match; re-downloading the file'
                )
    with urllib.request.urlopen(url) as source, open(download_target, 'wb'
        ) as output:
        with tqdm(total=int(source.info().get('Content-Length')), ncols=80,
            unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if insecure_hashlib.sha256(open(download_target, 'rb').read()).hexdigest(
        ) != expected_sha256:
        raise RuntimeError(
            'Model has been downloaded but the SHA256 checksum does not not match'
            )
    return download_target
