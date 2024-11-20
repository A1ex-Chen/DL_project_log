def git_describe(path=ROOT):
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always',
            shell=True).decode()[:-1]
    except Exception:
        return ''
