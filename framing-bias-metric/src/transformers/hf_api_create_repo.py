def create_repo(self, token: str, name: str, organization: Optional[str]=None
    ) ->str:
    """
        HuggingFace git-based system, used for models.

        Call HF API to create a whole repo.
        """
    path = '{}/api/repos/create'.format(self.endpoint)
    r = requests.post(path, headers={'authorization': 'Bearer {}'.format(
        token)}, json={'name': name, 'organization': organization})
    r.raise_for_status()
    d = r.json()
    return d['url']
