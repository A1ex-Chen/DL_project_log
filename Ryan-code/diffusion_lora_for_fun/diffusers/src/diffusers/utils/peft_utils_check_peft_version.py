def check_peft_version(min_version: str) ->None:
    """
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    if not is_peft_available():
        raise ValueError(
            'PEFT is not installed. Please install it with `pip install peft`')
    is_peft_version_compatible = version.parse(importlib.metadata.version(
        'peft')) > version.parse(min_version)
    if not is_peft_version_compatible:
        raise ValueError(
            f'The version of PEFT you are using is not compatible, please use a version that is greater than {min_version}'
            )
