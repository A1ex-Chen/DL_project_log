@classmethod
def check_version_valid(cls, version: str):
    if version:
        major_version, minor_version = cls._parse_version_str(version)
        cls._check_version_number(major_version)
        cls._check_version_number(minor_version)
        cls._check_major_version(major_version, minor_version)
