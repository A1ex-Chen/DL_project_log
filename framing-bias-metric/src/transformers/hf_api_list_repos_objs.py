def list_repos_objs(self, token: str, organization: Optional[str]=None) ->List[
    S3Obj]:
    """
        HuggingFace git-based system, used for models.

        Call HF API to list all stored files for user (or one of their organizations).
        """
    path = '{}/api/repos/ls'.format(self.endpoint)
    params = {'organization': organization
        } if organization is not None else None
    r = requests.get(path, params=params, headers={'authorization':
        'Bearer {}'.format(token)})
    r.raise_for_status()
    d = r.json()
    return [RepoObj(**x) for x in d]
