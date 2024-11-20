def is_url(url, check_online=True):
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc, result.path])
        return urllib.request.urlopen(url).getcode(
            ) == 200 if check_online else True
    except (AssertionError, urllib.request.HTTPError):
        return False
