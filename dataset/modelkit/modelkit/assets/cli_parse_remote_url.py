def parse_remote_url(path):
    match = re.match(storage_url_re, path)
    if not match:
        raise ValueError(f'Could not parse path `{path}`')
    return match.groupdict()
