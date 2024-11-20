def get_identifier(filepath):
    identifier = os.path.splitext(filepath)[0]
    to_replace = os.path.dirname(os.path.dirname(os.path.dirname(identifier)))
    return identifier.replace(to_replace, '')
