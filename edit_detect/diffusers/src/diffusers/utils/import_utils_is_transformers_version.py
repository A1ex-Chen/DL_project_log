def is_transformers_version(operation: str, version: str):
    """
    Args:
    Compares the current Transformers version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _transformers_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)
