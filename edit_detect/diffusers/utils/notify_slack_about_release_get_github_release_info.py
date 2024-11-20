def get_github_release_info(github_repo):
    """Fetch the latest release info from GitHub."""
    url = f'https://api.github.com/repos/{github_repo}/releases/latest'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {'tag_name': data['tag_name'], 'url': data['html_url'],
            'release_time': data['published_at']}
    else:
        print('Failed to fetch release info from GitHub.')
        return None
