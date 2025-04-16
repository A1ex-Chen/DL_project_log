def require_version(requirement: str, hint: Optional[str]=None) ->None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the `site-packages` dir via `pkg_resources`.

    Args:
        requirement (:obj:`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (:obj:`str`, `optional`): what suggestion to print in case of requirements not being met
    """
    hint = f'\n{hint}' if hint is not None else ''
    if re.match('^[\\w_\\-\\d]+$', requirement):
        pkg, op, want_ver = requirement, None, None
    else:
        match = re.findall('^([^!=<>\\s]+)([\\s!=<>]{1,2})(.+)', requirement)
        if not match:
            raise ValueError(
                f'requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}'
                )
        pkg, op, want_ver = match[0]
        if op not in ops:
            raise ValueError(f'need one of {list(ops.keys())}, but got {op}')
    if pkg == 'python':
        got_ver = '.'.join([str(x) for x in sys.version_info[:3]])
        if not ops[op](version.parse(got_ver), version.parse(want_ver)):
            raise pkg_resources.VersionConflict(
                f'{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.'
                )
        return
    try:
        got_ver = pkg_resources.get_distribution(pkg).version
    except pkg_resources.DistributionNotFound:
        raise pkg_resources.DistributionNotFound(requirement, [
            'this application', hint])
    if want_ver is not None and not ops[op](version.parse(got_ver), version
        .parse(want_ver)):
        raise pkg_resources.VersionConflict(
            f'{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}'
            )
