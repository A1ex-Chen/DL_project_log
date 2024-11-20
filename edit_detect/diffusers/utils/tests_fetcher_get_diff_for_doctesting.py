def get_diff_for_doctesting(repo: Repo, base_commit: str, commits: List[str]
    ) ->List[str]:
    """
    Get the diff in doc examples between a base commit and one or several commits.

    Args:
        repo (`git.Repo`):
            A git repository (for instance the Transformers repo).
        base_commit (`str`):
            The commit reference of where to compare for the diff. This is the current commit, not the branching point!
        commits (`List[str]`):
            The list of commits with which to compare the repo at `base_commit` (so the branching point).

    Returns:
        `List[str]`: The list of Python and Markdown files with a diff (files added or renamed are always returned, files
        modified are returned if the diff in the file is only in doctest examples).
    """
    print('\n### DIFF ###\n')
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            if not diff_obj.b_path.endswith('.py'
                ) and not diff_obj.b_path.endswith('.md'):
                continue
            if diff_obj.change_type in ['A']:
                code_diff.append(diff_obj.b_path)
            elif diff_obj.change_type in ['M', 'R']:
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                elif diff_contains_doc_examples(repo, commit, diff_obj.b_path):
                    code_diff.append(diff_obj.a_path)
                else:
                    print(
                        f"Ignoring diff in {diff_obj.b_path} as it doesn't contain any doc example."
                        )
    return code_diff
