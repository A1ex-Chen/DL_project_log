@classmethod
def parse_semantic_version(cls, version_str):
    if VERSION_REGEX.match(version_str) is None:
        return None
    version_nums = list(map(int, version_str.split('.')))
    return cls(major=version_nums[0], minor=version_nums[1], patch=
        version_nums[2])
