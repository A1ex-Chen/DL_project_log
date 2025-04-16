def is_peft_version(operation: str, version: str):
    """
    Args:
    Compares the current PEFT version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _peft_version:
        return False
    return compare_versions(parse(_peft_version), operation, version)
