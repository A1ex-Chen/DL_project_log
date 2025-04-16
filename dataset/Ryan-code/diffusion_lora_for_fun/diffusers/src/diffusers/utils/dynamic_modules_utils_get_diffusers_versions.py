def get_diffusers_versions():
    url = 'https://pypi.org/pypi/diffusers/json'
    releases = json.loads(request.urlopen(url).read())['releases'].keys()
    return sorted(releases, key=lambda x: version.Version(x))
