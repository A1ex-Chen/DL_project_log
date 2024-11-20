def github_assets(repository, version='latest'):
    if version != 'latest':
        version = f'tags/{version}'
    response = requests.get(
        f'https://api.github.com/repos/{repository}/releases/{version}').json()
    return response['tag_name'], [x['name'] for x in response['assets']]
