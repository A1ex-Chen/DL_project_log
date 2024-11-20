@classmethod
@abc.abstractmethod
def check_version_valid(cls, version: str):
    """raises InvalidVersionError if version is not valid"""
    ...
