def is_url(url):
    """ Checks if input string is a URL.

    Args:
        url (string): URL
    """
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
