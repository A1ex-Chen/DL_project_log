def fetch_all_branches(user, repo):
    branches = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{user}/{repo}/branches', params=
            {'page': page})
        if response.status_code == 200:
            branches.extend([branch['name'] for branch in response.json()])
            if 'next' in response.links:
                page += 1
            else:
                break
        else:
            print('Failed to retrieve branches:', response.status_code)
            break
    return branches
