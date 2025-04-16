def require_peft_version_greater(peft_version):
    """
    Decorator marking a test that requires PEFT backend with a specific version, this would require some specific
    versions of PEFT and transformers.
    """

    def decorator(test_case):
        correct_peft_version = is_peft_available() and version.parse(version
            .parse(importlib.metadata.version('peft')).base_version
            ) > version.parse(peft_version)
        return unittest.skipUnless(correct_peft_version,
            f'test requires PEFT backend with the version greater than {peft_version}'
            )(test_case)
    return decorator
