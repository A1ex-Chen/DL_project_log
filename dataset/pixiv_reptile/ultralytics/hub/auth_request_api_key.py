def request_api_key(self, max_attempts=3):
    """
        Prompt the user to input their API key.

        Returns the model ID.
        """
    import getpass
    for attempts in range(max_attempts):
        LOGGER.info(f'{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}')
        input_key = getpass.getpass(f'Enter API key from {API_KEY_URL} ')
        self.api_key = input_key.split('_')[0]
        if self.authenticate():
            return True
    raise ConnectionError(emojis(f'{PREFIX}Failed to authenticate ❌'))
