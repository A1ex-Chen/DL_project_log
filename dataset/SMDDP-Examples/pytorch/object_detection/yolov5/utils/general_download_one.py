def download_one(url, dir):
    success = True
    f = dir / Path(url).name
    if Path(url).is_file():
        Path(url).rename(f)
    elif not f.exists():
        LOGGER.info(f'Downloading {url} to {f}...')
        for i in range(retry + 1):
            if curl:
                s = 'sS' if threads > 1 else ''
                r = os.system(f'curl -{s}L "{url}" -o "{f}" --retry 9 -C -')
                success = r == 0
            else:
                torch.hub.download_url_to_file(url, f, progress=threads == 1)
                success = f.is_file()
            if success:
                break
            elif i < retry:
                LOGGER.warning(
                    f'Download failure, retrying {i + 1}/{retry} {url}...')
            else:
                LOGGER.warning(f'Failed to download {url}...')
    if unzip and success and f.suffix in ('.zip', '.gz'):
        LOGGER.info(f'Unzipping {f}...')
        if f.suffix == '.zip':
            ZipFile(f).extractall(path=dir)
        elif f.suffix == '.gz':
            os.system(f'tar xfz {f} --directory {f.parent}')
        if delete:
            f.unlink()
