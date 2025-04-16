def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(':/', '://')
    return urllib.parse.unquote(url).split('?', maxsplit=1)[0]
