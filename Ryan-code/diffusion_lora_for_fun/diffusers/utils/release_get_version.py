def get_version():
    """Reads the current version in the __init__."""
    with open(REPLACE_FILES['init'], 'r') as f:
        code = f.read()
    default_version = REPLACE_PATTERNS['init'][0].search(code).groups()[0]
    return packaging.version.parse(default_version)
