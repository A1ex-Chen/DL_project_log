def download_one(url, dir):
    success = True
    if os.path.isfile(url):
        f = Path(url)
    else:
        f = dir / Path(url).name
        LOGGER.info(f'Downloading {url} to {f}...')
        for i in range(retry + 1):
            if curl:
                success = curl_download(url, f, silent=threads > 1)
            else:
                torch.hub.download_url_to_file(url, f, progress=threads == 1)
                success = f.is_file()
            if success:
                break
            if i < retry:
                LOGGER.warning(
                    f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')
            else:
                LOGGER.warning(f'❌ Failed to download {url}...')
    if unzip and success and (f.suffix == '.gz' or is_zipfile(f) or
        is_tarfile(f)):
        LOGGER.info(f'Unzipping {f}...')
        if is_zipfile(f):
            unzip_file(f, dir)
        elif is_tarfile(f):
            subprocess.run(['tar', 'xf', f, '--directory', f.parent], check
                =True)
        elif f.suffix == '.gz':
            subprocess.run(['tar', 'xfz', f, '--directory', f.parent],
                check=True)
        if delete:
            f.unlink()
