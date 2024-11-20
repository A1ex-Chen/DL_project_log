def is_accelerate_version(operation: str, version: str):
    """
    Args:
    Compares the current Accelerate version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _accelerate_available:
        return False
    return compare_versions(parse(_accelerate_version), operation, version)
