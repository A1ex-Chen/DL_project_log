def is_url(url):
    """ Checks if input is url."""
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
