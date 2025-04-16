def require_accelerate_version_greater(accelerate_version):

    def decorator(test_case):
        correct_accelerate_version = is_peft_available() and version.parse(
            version.parse(importlib.metadata.version('accelerate')).
            base_version) > version.parse(accelerate_version)
        return unittest.skipUnless(correct_accelerate_version,
            f'Test requires accelerate with the version greater than {accelerate_version}.'
            )(test_case)
    return decorator
