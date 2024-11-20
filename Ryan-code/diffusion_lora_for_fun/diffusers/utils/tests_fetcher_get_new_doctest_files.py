def get_new_doctest_files(repo, base_commit, branching_commit) ->List[str]:
    """
    Get the list of files that were removed from "utils/not_doctested.txt", between `base_commit` and
    `branching_commit`.

    Returns:
        `List[str]`: List of files that were removed from "utils/not_doctested.txt".
    """
    for diff_obj in branching_commit.diff(base_commit):
        if diff_obj.a_path != 'utils/not_doctested.txt':
            continue
        folder = Path(repo.working_dir)
        with checkout_commit(repo, branching_commit):
            with open(folder / 'utils/not_doctested.txt', 'r', encoding='utf-8'
                ) as f:
                old_content = f.read()
        with open(folder / 'utils/not_doctested.txt', 'r', encoding='utf-8'
            ) as f:
            new_content = f.read()
        removed_content = {x.split(' ')[0] for x in old_content.split('\n')
            } - {x.split(' ')[0] for x in new_content.split('\n')}
        return sorted(removed_content)
    return []
