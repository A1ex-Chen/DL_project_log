def require_version(requirement: str, hint: Optional[str]=None) ->None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the *site-packages* dir via *importlib.metadata*.

    Args:
        requirement (`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (`str`, *optional*): what suggestion to print in case of requirements not being met

    Example:

    ```python
    require_version("pandas>1.1.2")
    require_version("numpy>1.18.5", "this is important to have for whatever reason")
    ```"""
    hint = f'\n{hint}' if hint is not None else ''
    if re.match('^[\\w_\\-\\d]+$', requirement):
        pkg, op, want_ver = requirement, None, None
    else:
        match = re.findall('^([^!=<>\\s]+)([\\s!=<>]{1,2}.+)', requirement)
        if not match:
            raise ValueError(
                f'requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}'
                )
        pkg, want_full = match[0]
        want_range = want_full.split(',')
        wanted = {}
        for w in want_range:
            match = re.findall('^([\\s!=<>]{1,2})(.+)', w)
            if not match:
                raise ValueError(
                    f'requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}'
                    )
            op, want_ver = match[0]
            wanted[op] = want_ver
            if op not in ops:
                raise ValueError(
                    f'{requirement}: need one of {list(ops.keys())}, but got {op}'
                    )
    if pkg == 'python':
        got_ver = '.'.join([str(x) for x in sys.version_info[:3]])
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return
    try:
        got_ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        raise importlib.metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
            )
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
