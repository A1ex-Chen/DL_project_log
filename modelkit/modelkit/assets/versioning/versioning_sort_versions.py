@classmethod
@abc.abstractmethod
def sort_versions(cls, version_list: typing.Iterable[str]) ->typing.List[str]:
    """Sort the version_list according to the versioning system"""
    ...
