@staticmethod
def _parse_version_str(version: str):
    m = re.fullmatch(MAJOR_MINOR_VERSION_RE, version)
    if not m:
        raise errors.InvalidVersionError(version)
    d = m.groupdict()
    return d['major'], d['minor'] if d.get('minor') else None
