@classmethod
def get_latest_partial_version(cls, version: str, versions: typing.List[str]
    ) ->str:
    """Override for a system with partial / incomplete version"""
    return versions[0]
