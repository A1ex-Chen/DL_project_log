def compare_versions(library_or_version: Union[str, Version], operation:
    str, requirement_version: str):
    """
    Args:
    Compares a library version to some requirement using a given operation.
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(
            f'`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}'
            )
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(
            library_or_version))
    return operation(library_or_version, parse(requirement_version))
