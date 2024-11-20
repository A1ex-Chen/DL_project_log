def get_system_metadata(repo_root):
    import git
    return dict(helsinki_git_sha=git.Repo(path=repo_root,
        search_parent_directories=True).head.object.hexsha,
        transformers_git_sha=git.Repo(path='.', search_parent_directories=
        True).head.object.hexsha, port_machine=socket.gethostname(),
        port_time=time.strftime('%Y-%m-%d-%H:%M'))
