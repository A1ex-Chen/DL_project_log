def authenticate(self) ->bool:
    """
        Attempt to authenticate with the server using either id_token or API key.

        Returns:
            (bool): True if authentication is successful, False otherwise.
        """
    try:
        if (header := self.get_auth_header()):
            r = requests.post(f'{HUB_API_ROOT}/v1/auth', headers=header)
            if not r.json().get('success', False):
                raise ConnectionError('Unable to authenticate.')
            return True
        raise ConnectionError('User has not authenticated locally.')
    except ConnectionError:
        self.id_token = self.api_key = False
        LOGGER.warning(f'{PREFIX}Invalid API key ⚠️')
        return False
