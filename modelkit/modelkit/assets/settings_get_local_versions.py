def get_local_versions(self, local_name) ->typing.List[str]:
    if os.path.isdir(local_name):
        return self.versioning.sort_versions(version_list=[d for d in os.
            listdir(local_name) if self.versioning.is_version_valid(d)])
    else:
        return []
