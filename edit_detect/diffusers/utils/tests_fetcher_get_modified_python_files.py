def get_modified_python_files(diff_with_last_commit: bool=False) ->List[str]:
    """
    Return a list of python files that have been modified between:

    - the current head and the main branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.

    Returns:
        `List[str]`: The list of Python files with a diff (files added, renamed or deleted are always returned, files
        modified are returned if the diff in the file is not only in docstrings or comments, see
        `diff_is_docstring_only`).
    """
    repo = Repo(PATH_TO_REPO)
    if not diff_with_last_commit:
        upstream_main = repo.remotes.origin.refs.main
        print(f'main is at {upstream_main.commit}')
        print(f'Current head is at {repo.head.commit}')
        branching_commits = repo.merge_base(upstream_main, repo.head)
        for commit in branching_commits:
            print(f'Branching commit: {commit}')
        return get_diff(repo, repo.head.commit, branching_commits)
    else:
        print(f'main is at {repo.head.commit}')
        parent_commits = repo.head.commit.parents
        for commit in parent_commits:
            print(f'Parent commit: {commit}')
        return get_diff(repo, repo.head.commit, parent_commits)
