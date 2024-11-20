def _get_failure_message(self, response: requests.Response, retry: int,
    timeout: int):
    """
        Generate a retry message based on the response status code.

        Args:
            response: The HTTP response object.
            retry: The number of retry attempts allowed.
            timeout: The maximum timeout duration.

        Returns:
            (str): The retry message.
        """
    if self._should_retry(response.status_code):
        return f'Retrying {retry}x for {timeout}s.' if retry else ''
    elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        headers = response.headers
        return (
            f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). Please retry after {headers['Retry-After']}s."
            )
    else:
        try:
            return response.json().get('message', 'No JSON message.')
        except AttributeError:
            return 'Unable to read JSON.'
