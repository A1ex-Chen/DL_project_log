@classmethod
def check_version_valid(cls, version: str):
    if not re.fullmatch(DATE_RE, version):
        raise errors.InvalidVersionError(f'Invalid version `{version}`')
