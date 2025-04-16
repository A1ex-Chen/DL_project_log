@classmethod
def sort_versions(cls, version_list: typing.Iterable[str]) ->typing.List[str]:
    return sorted(version_list, reverse=True)
