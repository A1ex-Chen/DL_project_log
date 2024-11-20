def get_datasets_dir():
    git_dir = get_git_dir()
    root = git_dir or Path()
    datasets_root = (root.parent if git_dir and is_dir_writeable(root.
        parent) else root).resolve()
    return datasets_root / 'datasets'
