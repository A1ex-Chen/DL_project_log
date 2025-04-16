def is_version_complete(self):
    if self.version:
        return self.versioning.is_version_complete(self.version)
    return False
