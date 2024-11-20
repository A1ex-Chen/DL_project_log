@classmethod
def latest_version(cls, version_list, major=None):
    if major:
        filtered_version_list = list(cls.filter_versions(version_list, major))
        if not filtered_version_list:
            raise MajorVersionDoesNotExistError(major)
        return cls.sort_versions(filtered_version_list)[0]
    return cls.sort_versions(version_list)[0]
