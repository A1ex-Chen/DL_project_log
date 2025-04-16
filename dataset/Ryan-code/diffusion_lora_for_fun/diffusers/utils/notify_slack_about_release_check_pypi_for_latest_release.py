def check_pypi_for_latest_release(library_name):
    """Check PyPI for the latest release of the library."""
    response = requests.get(f'https://pypi.org/pypi/{library_name}/json')
    if response.status_code == 200:
        data = response.json()
        return data['info']['version']
    else:
        print('Failed to fetch library details from PyPI.')
        return None
