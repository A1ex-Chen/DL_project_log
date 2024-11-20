def post_release_work():
    """Do all the necessary post-release steps."""
    current_version = get_version()
    dev_version = f'{current_version.major}.{current_version.minor + 1}.0.dev0'
    current_version = current_version.base_version
    version = input(f'Which version are we developing now? [{dev_version}]')
    if len(version) == 0:
        version = dev_version
    print(f'Updating version to {version}.')
    global_version_update(version)
