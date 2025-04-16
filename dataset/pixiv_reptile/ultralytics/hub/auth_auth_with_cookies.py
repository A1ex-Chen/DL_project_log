def auth_with_cookies(self) ->bool:
    """
        Attempt to fetch authentication via cookies and set id_token. User must be logged in to HUB and running in a
        supported browser.

        Returns:
            (bool): True if authentication is successful, False otherwise.
        """
    if not IS_COLAB:
        return False
    try:
        authn = request_with_credentials(f'{HUB_API_ROOT}/v1/auth/auto')
        if authn.get('success', False):
            self.id_token = authn.get('data', {}).get('idToken', None)
            self.authenticate()
            return True
        raise ConnectionError('Unable to fetch browser authentication details.'
            )
    except ConnectionError:
        self.id_token = False
        return False
