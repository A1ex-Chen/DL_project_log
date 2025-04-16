def is_url(url, check=True):
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        return urllib.request.urlopen(url).getcode() == 200 if check else True
    except (AssertionError, urllib.request.HTTPError):
        return False
