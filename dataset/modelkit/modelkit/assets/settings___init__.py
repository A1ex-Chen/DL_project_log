def __init__(self, name: str, versioning: typing.Optional[str]=None,
    version: typing.Optional[str]=None, sub_part: typing.Optional[str]=None
    ) ->None:
    versioning = versioning or os.environ.get(
        'MODELKIT_ASSETS_VERSIONING_SYSTEM') or 'major_minor'
    if versioning == 'major_minor':
        self.versioning = MajorMinorAssetsVersioningSystem()
    elif versioning == 'simple_date':
        self.versioning = SimpleDateAssetsVersioningSystem()
    else:
        raise errors.UnknownAssetsVersioningSystemError(versioning)
    self.check_name_valid(name)
    self.name = name
    if version:
        self.check_version_valid(version)
        self.versioning.check_version_valid(version)
    self.version = version
    self.sub_part = sub_part
