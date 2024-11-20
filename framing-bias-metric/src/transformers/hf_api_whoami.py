def whoami(self, token: str) ->Tuple[str, List[str]]:
    """
        Call HF API to know "whoami"
        """
    path = '{}/api/whoami'.format(self.endpoint)
    r = requests.get(path, headers={'authorization': 'Bearer {}'.format(token)}
        )
    r.raise_for_status()
    d = r.json()
    return d['user'], d['orgs']
