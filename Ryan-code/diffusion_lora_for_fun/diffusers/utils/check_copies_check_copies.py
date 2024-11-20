def check_copies(overwrite: bool=False):
    all_files = glob.glob(os.path.join(DIFFUSERS_PATH, '**/*.py'),
        recursive=True)
    diffs = []
    for filename in all_files:
        new_diffs = is_copy_consistent(filename, overwrite)
        diffs += [f'- {filename}: copy does not match {d[0]} at line {d[1]}'
             for d in new_diffs]
    if not overwrite and len(diffs) > 0:
        diff = '\n'.join(diffs)
        raise Exception('Found the following copy inconsistencies:\n' +
            diff +
            """
Run `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them."""
            )
