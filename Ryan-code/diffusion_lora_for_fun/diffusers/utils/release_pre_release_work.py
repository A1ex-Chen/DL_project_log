def pre_release_work(patch=False):
    """Do all the necessary pre-release steps."""
    default_version = get_version()
    if patch and default_version.is_devrelease:
        raise ValueError(
            "Can't create a patch version from the dev branch, checkout a released version!"
            )
    if default_version.is_devrelease:
        default_version = default_version.base_version
    elif patch:
        default_version = (
            f'{default_version.major}.{default_version.minor}.{default_version.micro + 1}'
            )
    else:
        default_version = (
            f'{default_version.major}.{default_version.minor + 1}.0')
    version = input(f'Which version are you releasing? [{default_version}]')
    if len(version) == 0:
        version = default_version
    print(f'Updating version to {version}.')
    global_version_update(version, patch=patch)
