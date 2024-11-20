@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo='ultralytics/yolov5', branch='master'):
    url = f'https://github.com/{repo}'
    msg = f', for updates see {url}'
    s = colorstr('github: ')
    assert Path('.git').exists(
        ), s + 'skipping check (not a git repository)' + msg
    assert check_online(), s + 'skipping check (offline)' + msg
    splits = re.split(pattern='\\s', string=check_output('git remote -v',
        shell=True).decode())
    matches = [(repo in s) for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = 'ultralytics'
        check_output(f'git remote add {remote} {url}', shell=True)
    check_output(f'git fetch {remote}', shell=True, timeout=5)
    local_branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True
        ).decode().strip()
    n = int(check_output(
        f'git rev-list {local_branch}..{remote}/{branch} --count', shell=True))
    if n > 0:
        pull = ('git pull' if remote == 'origin' else
            f'git pull {remote} {branch}')
        s += (
            f"⚠️ YOLOv3 is out of date by {n} commit{'s' * (n > 1)}. Use '{pull}' or 'git clone {url}' to update."
            )
    else:
        s += f'up to date with {url} ✅'
    LOGGER.info(s)
