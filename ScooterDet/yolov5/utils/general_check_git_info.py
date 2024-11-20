@WorkingDirectory(ROOT)
def check_git_info(path='.'):
    check_requirements('gitpython')
    import git
    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace('.git', '')
        commit = repo.head.commit.hexsha
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = None
        return {'remote': remote, 'branch': branch, 'commit': commit}
    except git.exc.InvalidGitRepositoryError:
        return {'remote': None, 'branch': None, 'commit': None}
