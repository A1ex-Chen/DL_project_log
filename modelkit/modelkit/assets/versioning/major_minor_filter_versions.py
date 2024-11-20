@staticmethod
def filter_versions(version_list, major):
    if not re.fullmatch('[0-9]+', major):
        raise InvalidMajorVersionError(major)
    return [v for v in version_list if re.match(f'^{major}' + '.', v)]
