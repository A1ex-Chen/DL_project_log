def login(self, username: str, password: str) ->str:
    """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs: token if credentials are valid

        Throws: requests.exceptions.HTTPError if credentials are invalid
        """
    path = '{}/api/login'.format(self.endpoint)
    r = requests.post(path, json={'username': username, 'password': password})
    r.raise_for_status()
    d = r.json()
    return d['token']
