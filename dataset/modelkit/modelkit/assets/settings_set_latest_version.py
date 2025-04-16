def set_latest_version(self, all_versions: typing.List[str]):
    if not self.version:
        self.version = all_versions[0]
    else:
        self.version = self.versioning.get_latest_partial_version(self.
            version, all_versions)
