def url2file(url):
    url = str(Path(url)).replace(':/', '://')
    return Path(urllib.parse.unquote(url)).name.split('?')[0]
