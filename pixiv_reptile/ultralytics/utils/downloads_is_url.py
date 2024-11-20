def is_url(url, check=False):
    """
    Validates if the given string is a URL and optionally checks if the URL exists online.

    Args:
        url (str): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.
            Defaults to True.

    Returns:
        (bool): Returns True for a valid URL. If 'check' is True, also returns True if the URL exists online.
            Returns False otherwise.

    Example:
        ```python
        valid = is_url("https://www.example.com")
        ```
    """
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200
        return True
    return False
