@classmethod
def get_latest_partial_version(cls, version: str, versions: typing.List[str]
    ) ->str:
    major_version, _ = cls._parse_version_str(version)
    if major_version:
        return cls.filter_versions(versions, major=major_version)[0]
    else:
        return versions[0]
