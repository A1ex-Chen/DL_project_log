def check_git_status():
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'
        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip(
            '.git')
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD',
            shell=True).decode().strip()
        n = int(subprocess.check_output(
            f'git rev-list {branch}..origin/master --count', shell=True))
        if n > 0:
            s = (
                f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. Use 'git pull' to update or 'git clone {url}' to download latest."
                )
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))
    except Exception as e:
        print(e)
