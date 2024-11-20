@classmethod
def is_version_complete(cls, version: str):
    try:
        major_version, minor_version = cls._parse_version_str(version)
    except InvalidMajorVersionError:
        return False
    return major_version is not None and minor_version is not None
