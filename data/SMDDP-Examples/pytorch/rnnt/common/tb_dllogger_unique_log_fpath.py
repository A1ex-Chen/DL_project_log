def unique_log_fpath(log_fpath):
    if not os.path.isfile(log_fpath):
        return log_fpath
    saved = sorted([int(re.search('\\.(\\d+)', f).group(1)) for f in glob.
        glob(f'{log_fpath}.*')])
    log_num = (saved[-1] if saved else 0) + 1
    return f'{log_fpath}.{log_num}'
