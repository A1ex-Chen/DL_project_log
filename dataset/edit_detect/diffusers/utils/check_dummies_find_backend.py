def find_backend(line):
    """Find one (or multiple) backend in a code line of the init."""
    backends = _re_backend.findall(line)
    if len(backends) == 0:
        return None
    return '_and_'.join(backends)
