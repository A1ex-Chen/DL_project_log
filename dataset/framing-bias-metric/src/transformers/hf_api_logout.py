def logout(self, token: str) ->None:
    """
        Call HF API to log out.
        """
    path = '{}/api/logout'.format(self.endpoint)
    r = requests.post(path, headers={'authorization': 'Bearer {}'.format(
        token)})
    r.raise_for_status()
