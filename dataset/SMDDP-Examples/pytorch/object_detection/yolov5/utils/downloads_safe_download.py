def safe_download(file, url, url2=None, min_bytes=1.0, error_msg=''):
    from utils.general import LOGGER
    file = Path(file)
    assert_msg = (
        f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
        )
    try:
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.
            level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg
    except Exception as e:
        file.unlink(missing_ok=True)
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            file.unlink(missing_ok=True)
            LOGGER.info(f'ERROR: {assert_msg}\n{error_msg}')
        LOGGER.info('')
