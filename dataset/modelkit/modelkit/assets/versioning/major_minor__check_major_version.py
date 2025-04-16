@staticmethod
def _check_major_version(major_version, minor_version):
    if minor_version and not major_version:
        raise errors.InvalidVersionError(
            'Cannot specify a minor version without a major version.')
