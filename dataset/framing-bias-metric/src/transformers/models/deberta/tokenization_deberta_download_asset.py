def download_asset(name, tag=None, no_cache=False, cache_dir=None):
    _tag = tag
    if _tag is None:
        _tag = 'latest'
    if not cache_dir:
        cache_dir = os.path.join(pathlib.Path.home(),
            f'.~DeBERTa/assets/{_tag}/')
    os.makedirs(cache_dir, exist_ok=True)
    output = os.path.join(cache_dir, name)
    if os.path.exists(output) and not no_cache:
        return output
    repo = 'https://api.github.com/repos/microsoft/DeBERTa/releases'
    releases = requests.get(repo).json()
    if tag and tag != 'latest':
        release = [r for r in releases if r['name'].lower() == tag.lower()]
        if len(release) != 1:
            raise Exception(f"{tag} can't be found in the repository.")
    else:
        release = releases[0]
    asset = [s for s in release['assets'] if s['name'].lower() == name.lower()]
    if len(asset) != 1:
        raise Exception(f"{name} can't be found in the release.")
    url = asset[0]['url']
    headers = {}
    headers['Accept'] = 'application/octet-stream'
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code != 200:
        raise Exception(
            f'Request for {url} return {resp.status_code}, {resp.text}')
    try:
        with open(output, 'wb') as fs:
            progress = tqdm(total=int(resp.headers['Content-Length']) if 
                'Content-Length' in resp.headers else -1, ncols=80, desc=
                f'Downloading {name}')
            for c in resp.iter_content(chunk_size=1024 * 1024):
                fs.write(c)
            progress.update(len(c))
            progress.close()
    except Exception:
        os.remove(output)
        raise
    return output
