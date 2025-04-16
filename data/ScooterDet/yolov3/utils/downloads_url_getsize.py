def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))
