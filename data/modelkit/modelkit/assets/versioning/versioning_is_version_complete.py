@classmethod
def is_version_complete(cls, version: str) ->bool:
    """Override for a system with partial / incomplete version"""
    return True
