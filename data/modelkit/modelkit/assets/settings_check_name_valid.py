@classmethod
def check_name_valid(cls, name: str):
    if not re.fullmatch(GENERIC_ASSET_NAME_RE, name):
        raise errors.InvalidNameError(
            f'Invalid name `{name}`, can only contain [a-z], [0-9], [/], [-] or [_]'
            )
