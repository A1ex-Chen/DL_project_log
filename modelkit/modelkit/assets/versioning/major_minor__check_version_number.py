@staticmethod
def _check_version_number(minor_or_major):
    if minor_or_major and not re.fullmatch('^[0-9]+$', minor_or_major):
        raise errors.InvalidVersionError(
            f'Invalid version `{minor_or_major}` is not a number')
