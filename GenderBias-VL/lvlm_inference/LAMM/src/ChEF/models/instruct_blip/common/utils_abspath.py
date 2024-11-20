def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile('^\\w+://')
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path
