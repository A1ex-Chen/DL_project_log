def get_github_assets(repo='ultralytics/assets', version='latest', retry=False
    ):
    """
    Retrieve the specified version's tag and assets from a GitHub repository. If the version is not specified, the
    function fetches the latest release assets.

    Args:
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        version (str, optional): The release version to fetch assets from. Defaults to 'latest'.
        retry (bool, optional): Flag to retry the request in case of a failure. Defaults to False.

    Returns:
        (tuple): A tuple containing the release tag and a list of asset names.

    Example:
        ```python
        tag, assets = get_github_assets(repo='ultralytics/assets', version='latest')
        ```
    """
    if version != 'latest':
        version = f'tags/{version}'
    url = f'https://api.github.com/repos/{repo}/releases/{version}'
    r = requests.get(url)
    if r.status_code != 200 and r.reason != 'rate limit exceeded' and retry:
        r = requests.get(url)
    if r.status_code != 200:
        LOGGER.warning(
            f'⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}'
            )
        return '', []
    data = r.json()
    return data['tag_name'], [x['name'] for x in data['assets']]
