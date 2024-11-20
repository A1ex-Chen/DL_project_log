def delete_repo(self, token: str, name: str, organization: Optional[str]=None):
    """
        HuggingFace git-based system, used for models.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
    path = '{}/api/repos/delete'.format(self.endpoint)
    r = requests.delete(path, headers={'authorization': 'Bearer {}'.format(
        token)}, json={'name': name, 'organization': organization})
    r.raise_for_status()
