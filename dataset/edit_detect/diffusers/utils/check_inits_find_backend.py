def find_backend(line):
    """Find one (or multiple) backend in a code line of the init."""
    if _re_test_backend.search(line) is None:
        return None
    backends = [b[0] for b in _re_backend.findall(line)]
    backends.sort()
    return '_and_'.join(backends)
