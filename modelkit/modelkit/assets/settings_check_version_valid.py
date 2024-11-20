@classmethod
def check_version_valid(cls, name: str):
    if name and not re.fullmatch(GENERIC_ASSET_VERSION_RE, name):
        raise errors.InvalidVersionError(
            f'Invalid version `{name}`, can only contain [a-zA-Z0-9], [-._]')
