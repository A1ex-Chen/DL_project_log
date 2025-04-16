def download_ckpt(path):
    """Download checkpoints of the pretrained models"""
    basename = os.path.basename(path)
    dir = os.path.abspath(os.path.dirname(path))
    os.makedirs(dir, exist_ok=True)
    LOGGER.info(
        f'checkpoint {basename} not exist, try to downloaded it from github.')
    url = (
        f'https://github.com/meituan/YOLOv6/releases/download/0.4.0/{basename}'
        )
    LOGGER.warning(
        f'downloading url is: {url}, pealse make sure the version of the downloading model is correspoing to the code version!'
        )
    r = requests.get(url, allow_redirects=True)
    assert r.status_code == 200, 'Unable to download checkpoints, manually download it'
    open(path, 'wb').write(r.content)
    LOGGER.info(f'checkpoint {basename} downloaded and saved')
