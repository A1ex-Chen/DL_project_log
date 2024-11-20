@staticmethod
def from_string(input_string: str, versioning: typing.Optional[str]=None):
    match = re.match(REMOTE_ASSET_RE, input_string)
    if not match:
        raise errors.InvalidAssetSpecError(input_string)
    return AssetSpec(versioning=versioning, **match.groupdict())
