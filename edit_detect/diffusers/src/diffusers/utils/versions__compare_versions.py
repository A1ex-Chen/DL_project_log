def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None or want_ver is None:
        raise ValueError(
            f'Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This is unusual. Consider reinstalling {pkg}.'
            )
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f'{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}'
            )
