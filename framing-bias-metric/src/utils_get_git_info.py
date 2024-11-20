def get_git_info():
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_infos = {'repo_id': str(repo), 'repo_sha': str(repo.head.
            object.hexsha), 'repo_branch': str(repo.active_branch),
            'hostname': str(socket.gethostname())}
        return repo_infos
    except TypeError:
        return {'repo_id': None, 'repo_sha': None, 'repo_branch': None,
            'hostname': None}
