def is_version_valid(self, version: str) ->bool:
    try:
        self.check_version_valid(version)
        return True
    except errors.InvalidVersionError:
        return False
