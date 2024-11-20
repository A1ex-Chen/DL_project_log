def download_original_config(config_url, tmpdir):
    original_config_file = BytesIO(requests.get(config_url).content)
    path = f'{tmpdir}/config.yaml'
    with open(path, 'wb') as f:
        f.write(original_config_file.read())
    return path
