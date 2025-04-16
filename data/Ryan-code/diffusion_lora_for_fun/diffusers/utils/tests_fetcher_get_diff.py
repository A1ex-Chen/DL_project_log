def get_diff(repo: Repo, base_commit: str, commits: List[str]) ->List[str]:
    """
    Get the diff between a base commit and one or several commits.

    Args:
        repo (`git.Repo`):
            A git repository (for instance the Transformers repo).
        base_commit (`str`):
            The commit reference of where to compare for the diff. This is the current commit, not the branching point!
        commits (`List[str]`):
            The list of commits with which to compare the repo at `base_commit` (so the branching point).

    Returns:
        `List[str]`: The list of Python files with a diff (files added, renamed or deleted are always returned, files
        modified are returned if the diff in the file is not only in docstrings or comments, see
        `diff_is_docstring_only`).
    """
    print('\n### DIFF ###\n')
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            if diff_obj.change_type == 'A' and diff_obj.b_path.endswith('.py'):
                code_diff.append(diff_obj.b_path)
            elif diff_obj.change_type == 'D' and diff_obj.a_path.endswith('.py'
                ):
                code_diff.append(diff_obj.a_path)
            elif diff_obj.change_type in ['M', 'R'
                ] and diff_obj.b_path.endswith('.py'):
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                elif diff_is_docstring_only(repo, commit, diff_obj.b_path):
                    print(
                        f'Ignoring diff in {diff_obj.b_path} as it only concerns docstrings or comments.'
                        )
                else:
                    code_diff.append(diff_obj.a_path)
    return code_diff
