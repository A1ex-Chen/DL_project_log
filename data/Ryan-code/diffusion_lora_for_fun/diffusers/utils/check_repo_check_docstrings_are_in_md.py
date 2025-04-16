def check_docstrings_are_in_md():
    """Check all docstrings are in md"""
    files_with_rst = []
    for file in Path(PATH_TO_DIFFUSERS).glob('**/*.py'):
        with open(file, 'r') as f:
            code = f.read()
        docstrings = code.split('"""')
        for idx, docstring in enumerate(docstrings):
            if idx % 2 == 0 or not is_rst_docstring(docstring):
                continue
            files_with_rst.append(file)
            break
    if len(files_with_rst) > 0:
        raise ValueError(
            'The following files have docstrings written in rst:\n' + '\n'.
            join([f'- {f}' for f in files_with_rst]) +
            """
To fix this run `doc-builder convert path_to_py_file` after installing `doc-builder`
(`pip install git+https://github.com/huggingface/doc-builder`)"""
            )
