def get_config_name():
    import pkg_resources
    package_versions = {p.key: p.version for p in pkg_resources.working_set}
    return package_versions
