def save_git_info(folder_path: str) ->None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, 'git_log.json'))
