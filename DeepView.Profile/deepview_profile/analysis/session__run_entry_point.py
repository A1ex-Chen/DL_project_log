def _run_entry_point(path_to_entry_point, path_to_entry_point_dir, project_root
    ):
    with open(path_to_entry_point) as file:
        code_str = file.read()
    with exceptions_as_analysis_errors(project_root):
        tree = ast.parse(code_str, filename=path_to_entry_point)
        code = compile(tree, path_to_entry_point, mode='exec')
    with user_code_environment(path_to_entry_point_dir, project_root):
        scope = {}
        exec(code, scope, scope)
    return code_str, tree, scope
