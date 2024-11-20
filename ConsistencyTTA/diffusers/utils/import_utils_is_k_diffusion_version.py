def is_k_diffusion_version(operation: str, version: str):
    """
    Args:
    Compares the current k-diffusion version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _k_diffusion_available:
        return False
    return compare_versions(parse(_k_diffusion_version), operation, version)
