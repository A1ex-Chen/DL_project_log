def is_valid_url(url):
    result = urlparse(url)
    if result.scheme and result.netloc:
        return True
    return False
