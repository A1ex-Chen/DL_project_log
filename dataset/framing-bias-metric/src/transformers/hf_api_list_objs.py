def list_objs(self, token: str, organization: Optional[str]=None) ->List[S3Obj
    ]:
    """
        HuggingFace S3-based system, used for datasets and metrics.

        Call HF API to list all stored files for user (or one of their organizations).
        """
    path = '{}/api/datasets/listObjs'.format(self.endpoint)
    params = {'organization': organization
        } if organization is not None else None
    r = requests.get(path, params=params, headers={'authorization':
        'Bearer {}'.format(token)})
    r.raise_for_status()
    d = r.json()
    return [S3Obj(**x) for x in d]
