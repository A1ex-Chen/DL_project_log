def diff_contains_doc_examples(repo: Repo, branching_point: str, filename: str
    ) ->bool:
    """
    Check if the diff is only in code examples of the doc in a filename.

    Args:
        repo (`git.Repo`): A git repository (for instance the Transformers repo).
        branching_point (`str`): The commit reference of where to compare for the diff.
        filename (`str`): The filename where we want to know if the diff is only in codes examples.

    Returns:
        `bool`: Whether the diff is only in code examples of the doc or not.
    """
    folder = Path(repo.working_dir)
    with checkout_commit(repo, branching_point):
        with open(folder / filename, 'r', encoding='utf-8') as f:
            old_content = f.read()
    with open(folder / filename, 'r', encoding='utf-8') as f:
        new_content = f.read()
    old_content_clean = keep_doc_examples_only(old_content)
    new_content_clean = keep_doc_examples_only(new_content)
    return old_content_clean != new_content_clean
