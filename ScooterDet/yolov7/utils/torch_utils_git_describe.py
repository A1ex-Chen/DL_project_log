def git_describe(path=Path(__file__).parent):
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT
            ).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''
