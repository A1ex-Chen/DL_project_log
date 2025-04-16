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
        if file.exists():
            file.unlink()
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            if file.exists():
                file.unlink()
            LOGGER.info(f'ERROR: {assert_msg}\n{error_msg}')
        LOGGER.info('')
