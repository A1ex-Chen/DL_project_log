def get_auth_header(self):
    """
        Get the authentication header for making API requests.

        Returns:
            (dict): The authentication header if id_token or API key is set, None otherwise.
        """
    if self.id_token:
        return {'authorization': f'Bearer {self.id_token}'}
    elif self.api_key:
        return {'x-api-key': self.api_key}
