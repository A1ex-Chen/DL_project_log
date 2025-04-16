def git_describe(path=ROOT):
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    with contextlib.suppress(Exception):
        return subprocess.check_output(
            f'git -C {path} describe --tags --long --always', shell=True
            ).decode()[:-1]
    return ''
