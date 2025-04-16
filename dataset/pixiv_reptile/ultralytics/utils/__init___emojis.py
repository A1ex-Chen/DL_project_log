def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string
