@classmethod
def _parse_version(cls, version_str):
    major, minor = cls._parse_version_str(version_str)
    return int(major), int(minor) if minor is not None else None
