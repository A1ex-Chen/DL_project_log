def require_version_examples(requirement):
    """ require_version wrapper which emits examples-specific hint on failure """
    hint = 'Try: pip install -r examples/requirements.txt'
    return require_version(requirement, hint)
