def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(':/', '://')
    return urllib.parse.unquote(url).split('?')[0]
